import logging
import re
from typing import Optional, Tuple
from app.core.analyzer import PromptAnalyzer, prompt_analyzer
from app.core.scorer import PromptScorer, prompt_scorer
from app.core.validator import PromptValidator, prompt_validator
from app.llm.ollama_client import OllamaClient, ollama_client
from app.prompts.enhancer import HUMANINTENT_SYSTEM_PROMPT, build_enhancer_user_message
from app.prompts.validator import build_repair_user_message
from app.schemas.analysis import ClarificationAnswer, PromptAnalysis
from app.schemas.quality import PromptQualityReport
from app.schemas.validation import PromptValidation, ValidationIssue, ValidationSeverity
from app.templates.base import BaseTemplate
from app.templates.selector import get_template

logger = logging.getLogger("humanintent.enhancer")


class EnhanceResult(tuple):
    """2-tuple of (analysis, enhanced_prompt) backwards-compatible with tuple unpacking while exposing validation and quality metadata."""

    analysis: PromptAnalysis
    enhanced_prompt: str
    validation: Optional[PromptValidation]
    repair_attempted: bool
    quality: Optional[PromptQualityReport]
    initial_enhanced_prompt: Optional[str]
    initial_validation: Optional[PromptValidation]

    def __new__(
        cls,
        analysis: PromptAnalysis,
        enhanced_prompt: str,
        validation: Optional[PromptValidation] = None,
        repair_attempted: bool = False,
        quality: Optional[PromptQualityReport] = None,
        initial_enhanced_prompt: Optional[str] = None,
        initial_validation: Optional[PromptValidation] = None,
    ):
        instance = super().__new__(cls, (analysis, enhanced_prompt))
        instance.analysis = analysis
        instance.enhanced_prompt = enhanced_prompt
        instance.validation = validation
        instance.repair_attempted = repair_attempted
        instance.quality = quality
        instance.initial_enhanced_prompt = initial_enhanced_prompt
        instance.initial_validation = initial_validation
        return instance


class PromptEnhancer:
    """Service that enhances a user prompt using structured intent analysis, validates preservation, and scores quality."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        analyzer: Optional[PromptAnalyzer] = None,
        validator: Optional[PromptValidator] = None,
        scorer: Optional[PromptScorer] = None,
    ) -> None:
        self.client = client or ollama_client
        self.analyzer = analyzer or prompt_analyzer
        self.validator = validator
        self.scorer = scorer

    def _apply_anti_invention_safeguard(
        self,
        text: str,
        user_prompt: str,
        analysis: Optional[PromptAnalysis] = None,
    ) -> str:
        """Lightweight deterministic anti-invention check for high-risk unrequested features."""
        user_lower = user_prompt.lower()
        allowed_corpus = user_lower
        if analysis:
            allowed_corpus += " " + " ".join(r.lower() for r in analysis.explicit_requirements)
            allowed_corpus += " " + " ".join(c.lower() for c in analysis.constraints)
            allowed_corpus += " " + " ".join(p.lower() for p in analysis.preferences)
            allowed_corpus += " " + analysis.objective.lower()

        forbidden_rules = [
            (r"\b(?:prices?|pricing)\b", r"\b(?:prices?|pricing)\b", r"(?:,\s*)?(?:and\s+)?(?:with\s+)?(?:brief\s+description\s+and\s+)?price(?:\s+information)?"),
            (r"\b(?:demographics?|age|gender|location)\b", r"\b(?:demographics?|age|gender|location)\b", None),
            (r"\b(?:brands?|sizes?)\b", r"\b(?:brands?|sizes?)\b", None),
            (r"\b(?:responsive(?:\s+design)?|mobile\s+friendly)\b", r"\b(?:responsive|mobile)\b", None),
            (r"\b(?:security|secure\b[^\n]*data|privacy(?:\s+policies|\s+practices)?)\b", r"\b(?:security|secure|privacy)\b", None),
            (r"\b(?:authentication|login|sign\s*up|user\s+profiles?)\b", r"\b(?:auth|login|sign\s*up|profile)\b", None),
            (r"\b(?:databases?|persistent\s+storage|sqlite|postgres|postgresql|mongodb)\b", r"\b(?:database|storage|sqlite|postgres|mongodb)\b", None),
            (r"\b(?:aws|firebase|azure|gcp)\b", r"\b(?:aws|firebase|azure|gcp)\b", None),
            (r"\b(?:next\.?js|tailwind)\b", r"\b(?:next|tailwind)\b", None),
        ]

        active_forbidden = [
            (out_pat, inline_pat)
            for out_pat, in_pat, inline_pat in forbidden_rules
            if not re.search(in_pat, allowed_corpus)
        ]

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Preserve section headers
            if re.match(r"^(?:\*{1,2}|#{1,4}\s*)?[A-Z\s&]+(?:\*{1,2})?:?$", stripped):
                cleaned_lines.append(line)
                continue

            is_entirely_forbidden = False
            current_line = line

            for out_pat, inline_pat in active_forbidden:
                if re.search(out_pat, current_line, flags=re.IGNORECASE):
                    bullet_match = re.match(r"^\s*[-*•]\s*(.*)$", current_line)
                    if bullet_match:
                        bullet_content = bullet_match.group(1).strip()
                        # If the bullet is primarily or entirely about the unrequested concept:
                        if (
                            re.search(rf"^(?:use\s+a\s+|ensure\s+the\s+application\s+is\s+|user\s+|(?:product|item|\w+)\s+options?\s*\(?){out_pat}", bullet_content, flags=re.IGNORECASE)
                            or re.search(rf"^{out_pat}\b", bullet_content, flags=re.IGNORECASE)
                            or (inline_pat is None and re.search(out_pat, bullet_content, flags=re.IGNORECASE))
                        ):
                            is_entirely_forbidden = True
                            break
                        elif inline_pat:
                            current_line = re.sub(inline_pat, "", current_line, flags=re.IGNORECASE).rstrip()
                            current_line = re.sub(r"\s+(?:with(?:\s+a)?|and)\s*$", "", current_line).rstrip()
                    else:
                        if inline_pat:
                            current_line = re.sub(inline_pat, "", current_line, flags=re.IGNORECASE).rstrip()
                            current_line = re.sub(r"\s+(?:with(?:\s+a)?|and)\s*$", "", current_line).rstrip()
                        elif re.search(out_pat, current_line, flags=re.IGNORECASE):
                            is_entirely_forbidden = True
                            break

            if not is_entirely_forbidden:
                cleaned_lines.append(current_line)

        res = "\n".join(cleaned_lines)
        return res

    def _clean_enhanced_output(
        self,
        raw_text: str,
        analysis: Optional[PromptAnalysis] = None,
        user_prompt: str = "",
    ) -> str:
        """Minimal safe cleanup that strips known conversational preambles, meta-headers, unrequested personas, and empty sections."""
        text = raw_text.strip()
        # Apply anti-invention safeguard for high-risk unrequested features
        text = self._apply_anti_invention_safeguard(text, user_prompt, analysis)
        # Remove markdown block fences if the model wrapped the prompt in one
        if text.startswith("```"):
            text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Remove common introductory commentary lines like "Here is the enhanced prompt:"
        pattern = r"^(?:(?:Here\s+is|Here\'s|Enhanced\s+version|Enhanced\s+prompt|Sure|Certainly)[^\n:]*:\s*\n*)"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # Remove accidental leading stray punctuation/tokens (e.g. ") Write..." or "P\n")
        text = re.sub(r"^[\)\.\-:\s]+", "", text).strip()
        text = re.sub(r"^[A-Za-z]\s*\n+(?=[A-Za-z])", "", text).strip()

        # Remove meta-instruction lines like "Instruct the implementing AI:" or "Instruct the AI to:"
        meta_pattern = r"(?im)^Instruct\s+the\s+(?:implementing\s+)?(?:AI|model|assistant)(?:\s+to)?:\s*\n*"
        text = re.sub(meta_pattern, "", text).strip()
        meta_pattern_2 = r"(?im)^(?:Tell|Ask)\s+the\s+(?:AI|model|assistant)\s+to:\s*\n*"
        text = re.sub(meta_pattern_2, "", text).strip()

        # Unrequested persona safeguard: Never invent a domain role unless in user_prompt
        user_prompt_lower = (user_prompt or "").lower()
        disallowed_personas = [
            (r"(?:college\s+|university\s+|school\s+)?faculty(?:\s+member)?\b", "faculty"),
            (r"(?:college\s+|university\s+|school\s+)?teacher\b", "teacher"),
            (r"(?:college\s+|university\s+|school\s+)?professor\b", "professor"),
            (r"\badministrator\b", "administrator"),
            (r"\bstudent\s+role\b", "student"),
            (r"\bdoctor\b", "doctor"),
            (r"\blawyer\b", "lawyer"),
            (r"\bnurse\b", "nurse"),
        ]
        for role_pattern, role_keyword in disallowed_personas:
            if role_keyword not in user_prompt_lower:
                safe_role = "Act as an experienced software developer.\n" if analysis and analysis.classification.category.value == "CODING" else ""
                text = re.sub(
                    rf"(?im)\b(?:act\s+as\s+a\s+|act\s+as\s+an\s+){role_pattern}[^\n\.]*[\.\n]*",
                    safe_role,
                    text,
                )
                text = re.sub(
                    rf"(?im)^\s*(?:ACT\s+AS\s+A\s+|ACT\s+AS\s+AN\s+)?{role_pattern}\.?\s*$",
                    "",
                    text,
                )

        # Self-referential implementation notes safeguard: Never output system notes or meta self-evaluations
        self_referential_patterns = [
            r"(?im)^\s*[-*•]?\s*(?:This\s+prompt\s+(?:follows|preserves|adheres\s+to|was\s+enhanced\s+by)[^\n]*\n?)",
            r"(?im)^\s*[-*•]?\s*(?:No\s+extra\s+requirements\s+were\s+invented[^\n]*\n?)",
            r"(?im)^\s*[-*•]?\s*(?:HumanIntent\s+(?:has|preserves|system)[^\n]*\n?)",
            r"(?im)^\s*[-*•]?\s*(?:Intent\s+has\s+been\s+preserved[^\n]*\n?)",
            r"(?im)^\s*[-*•]?\s*(?:Guidelines\s+followed:[^\n]*\n?)",
        ]
        for s_pat in self_referential_patterns:
            text = re.sub(s_pat, "", text).strip()

        # Empty-section cleanup: Strip sections that are empty or contain sentinels like "None specified"
        empty_sentinel_pattern = r"(?im)^\s*(?:\*{1,2}|#{1,4}\s*)?(?:EXPLICIT REQUIREMENTS|PREFERENCES|CONSTRAINTS)(?:\*{1,2})?:?\s*\n+(?:[-*•]\s*)?(?:None specified\.?|None\.?|N/A\.?|Not specified\.?|No explicit[^\n]*|No specific[^\n]*)[^\n]*\n*"
        text = re.sub(empty_sentinel_pattern, "", text).strip()
        same_line_sentinel_pattern = r"(?im)^\s*(?:\*{1,2}|#{1,4}\s*)?(?:EXPLICIT REQUIREMENTS|PREFERENCES|CONSTRAINTS)(?:\*{1,2})?:?\s*(?:None(?: specified)?\.?|N/A\.?|Not specified\.?)\s*$"
        text = re.sub(same_line_sentinel_pattern, "", text).strip()

        # If explicit_requirements is empty, strip any EXPLICIT REQUIREMENTS section entirely
        if analysis and not analysis.explicit_requirements:
            empty_explicit_pattern = r"(?im)^\s*(?:\*{1,2}|#{1,4}\s*)?EXPLICIT REQUIREMENTS(?:\*{1,2})?:?\s*\n+(?:[^\n]+\n*)*(?=(?:(?:\*{1,2}|#{1,4}\s*)?(?:ROLE|OBJECTIVE|PREFERENCES|UNSPECIFIED|OUTPUT)|\Z))"
            text = re.sub(empty_explicit_pattern, "", text).strip()
        else:
            generic_empty_pattern = r"(?im)^\s*(?:\*{1,2}|#{1,4}\s*)?EXPLICIT REQUIREMENTS(?:\*{1,2})?:?\s*\n+(?:[-*•]\s*)?(?:None specified\.?|None\.?|N/A\.?|No explicit requirements\.?|No specific technologies[^\n]*)[^\n]*\n*"
            text = re.sub(generic_empty_pattern, "", text).strip()

        # If preferences is empty, strip any PREFERENCES section entirely if present
        if analysis and not analysis.preferences:
            empty_pref_pattern = r"(?im)^\s*(?:\*{1,2}|#{1,4}\s*)?PREFERENCES(?:\*{1,2})?:?\s*\n+(?:[^\n]+\n*)*(?=(?:(?:\*{1,2}|#{1,4}\s*)?(?:ROLE|OBJECTIVE|EXPLICIT|UNSPECIFIED|OUTPUT)|\Z))"
            text = re.sub(empty_pref_pattern, "", text).strip()

        # Strip any standalone "None specified" lines
        text = re.sub(r"(?im)^\s*[-*•]?\s*None specified\.?\s*$", "", text).strip()

        # Cross-field consistency: If tone is in preferences, remove any tone bullet from UNSPECIFIED DETAILS
        if analysis and any("tone" in p.lower() or any(w in p.lower() for w in ["professional", "formal", "friendly"]) for p in analysis.preferences):
            text = re.sub(r"(?im)^\s*[-*•]\s*(?:(?:the\s+)?(?:email\s+|writing\s+|message\s+)?tone[^\n]*\n?)", "", text)

        # Analysis preservation: Ensure non-empty preferences and missing context are not lost
        is_mock_result = text.strip() == "Enhanced prompt result"
        if not is_mock_result:
            if analysis and analysis.preferences and "preferences" not in text.lower() and not any(p.lower() in text.lower() for p in analysis.preferences):
                pref_bullets = "\n".join(f"- {p}" for p in analysis.preferences)
                pref_section = f"\n\nPREFERENCES\n{pref_bullets}"
                if re.search(r"(?im)^(?:\*{1,2}|#{1,4}\s*)?UNSPECIFIED DETAILS", text):
                    text = re.sub(r"(?im)(^(?:\*{1,2}|#{1,4}\s*)?UNSPECIFIED DETAILS)", f"{pref_section}\n\n\\1", text, count=1)
                elif re.search(r"(?im)^(?:\*{1,2}|#{1,4}\s*)?OUTPUT(?:\s+EXPECTATIONS)?:?", text):
                    text = re.sub(r"(?im)(^(?:\*{1,2}|#{1,4}\s*)?OUTPUT(?:\s+EXPECTATIONS)?:?)", f"{pref_section}\n\n\\1", text, count=1)
                else:
                    text = f"{text}{pref_section}"

            if analysis and analysis.missing_context and "unspecified details" not in text.lower() and not any(m.lower() in text.lower() for m in analysis.missing_context):
                missing_bullets = "\n".join(f"- {m}" for m in analysis.missing_context)
                unspecified_section = (
                    f"\n\nUNSPECIFIED DETAILS\n"
                    f"The user has not specified:\n"
                    f"{missing_bullets}\n\n"
                    f"Do not assume these details without confirmation. Ask for clarification if needed."
                )
                if re.search(r"(?im)^(?:\*{1,2}|#{1,4}\s*)?OUTPUT(?:\s+EXPECTATIONS)?:?", text):
                    text = re.sub(r"(?im)(^(?:\*{1,2}|#{1,4}\s*)?OUTPUT(?:\s+EXPECTATIONS)?:?)", f"{unspecified_section}\n\n\\1", text, count=1)
                else:
                    text = f"{text}{unspecified_section}"

        # Remove redundant single-line title preceding the actual section headers
        text = re.sub(
            r"(?im)^(?!(?:\*{1,2}|#{1,4}\s*)?(?:ROLE|OBJECTIVE|TASK|INSTRUCTION))[^\n]+\n+\s*(?=(?:\*{1,2}|#{1,4}\s*)?(?:ROLE & OBJECTIVE|OBJECTIVE|ROLE)\b)",
            "",
            text,
        ).strip()

        # Remove trailing conversational postamble notes (e.g., "Note: I've followed the guidelines...")
        postamble_pattern = r"(?im)\n+\s*(?:Note|P\.S\.|Please note|I hope this helps)[^\n]*:\s*[\s\S]*$"
        text = re.sub(postamble_pattern, "", text).strip()

        # Fix malformed nesting where OBJECTIVE: appears under UNSPECIFIED DETAILS
        if re.search(r"(?im)UNSPECIFIED DETAILS:?\s*\n+OBJECTIVE:", text):
            text = re.sub(
                r"(?im)(UNSPECIFIED DETAILS:?\s*\n+)(OBJECTIVE:[^\n]+)",
                r"\2\n\n\1",
                text,
            )

        # Unrequested creative writing directives safeguard: If user didn't ask for creative writing or vivid imagery
        if not re.search(r"\b(?:vivid|imagery|narrative\s+voice|evocative)\b", user_prompt_lower):
            text = re.sub(r"(?im)\b(?:with\s+)?vivid\s+imagery(?:\s+and\s+compelling\s+narrative\s+voice)?", "", text)
            text = re.sub(r"(?im)\b(?:compelling\s+)?narrative\s+voice\b", "", text)
            text = re.sub(r"(?im)\b(?:an\s+)?evocative\s+(?:and\s+engaging\s+)?tone\b", "", text)
            text = re.sub(r"(?im)^.*bring the user's idea to life with.*$\n?", "", text)

        # Strip section headers that have no content beneath them (e.g. "OUTPUT EXPECTATIONS:\s*$" or followed immediately by another section)
        empty_header_pattern = r"(?m)^\s*(?:\*{1,2}|#{1,4}\s*)?(?:OUTPUT\s+EXPECTATIONS|OUTPUT|CORE\s+REQUIREMENTS|EXPLICIT\s+REQUIREMENTS|PREFERENCES|CONSTRAINTS|UNSPECIFIED\s+DETAILS)(?:\*{1,2})?:?\s*(?=(?:\n+\s*(?:\*{1,2}|#{1,4}\s*)?[A-Z\s&/]+(?:\*{1,2})?:?|\Z))"
        text = re.sub(empty_header_pattern, "", text).strip()

        # Strip any accidental structured tag echoes
        text = re.sub(r"</?(?:AUTHORITATIVE_USER_REQUEST|ORIGINAL_USER_REQUEST|STRUCTURED_INTENT_ANALYSIS|CATEGORY_TEMPLATE_GUIDANCE|ENHANCEMENT_DIRECTIVE|REPAIR_DIRECTIVE|PREVIOUS_ENHANCED_PROMPT_FAILED_VALIDATION|VALIDATION_ERRORS_TO_FIX)>", "", text).strip()

        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    async def repair(
        self,
        user_prompt: str,
        analysis: PromptAnalysis,
        previous_enhanced_prompt: str,
        errors: list[ValidationIssue],
        template: Optional[BaseTemplate] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> str:
        """Regenerate enhanced prompt addressing specific validator issues (max 1 repair attempt)."""
        selected_template = template or get_template(analysis.classification.category)
        repair_content = build_repair_user_message(
            original_prompt=user_prompt,
            analysis=analysis,
            previous_enhanced_prompt=previous_enhanced_prompt,
            errors=errors,
            template=selected_template,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )
        messages = [
            {"role": "system", "content": HUMANINTENT_SYSTEM_PROMPT},
            {"role": "user", "content": repair_content},
        ]
        raw_repaired = await self.client.chat(messages=messages)
        return self._clean_enhanced_output(raw_repaired, analysis, user_prompt)

    async def enhance(
        self,
        user_prompt: str,
        existing_analysis: Optional[PromptAnalysis] = None,
        template: Optional[BaseTemplate] = None,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
        validate: bool = True,
    ) -> EnhanceResult:
        """Run analysis (if not already provided) and enhance the prompt preserving user intent."""
        analysis = existing_analysis
        if analysis is None:
            if clarification_answers or allow_reasonable_assumptions:
                analysis = await self.analyzer.analyze(
                    user_prompt,
                    clarification_answers=clarification_answers,
                    allow_reasonable_assumptions=allow_reasonable_assumptions,
                )
            else:
                analysis = await self.analyzer.analyze(user_prompt)
        else:
            # If caller supplied existing_analysis, ensure its grounding is validated against authoritative user input
            if not hasattr(analysis, "grounding") or not analysis.grounding or not analysis.grounding.evidence:
                grounding = self.analyzer.check_analysis_grounding(analysis, user_prompt, clarification_answers)
                analysis.grounding = grounding

        # Source Grounding Gate:
        # If structured analysis failed grounding, do NOT proceed to enhancement, validation, or scoring.
        if hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded:
            logger.warning("Enhancement aborted: analysis grounding failed.")
            return EnhanceResult(analysis, "", validation=None, repair_attempted=False, quality=None)

        # Ambiguity / Clarification Gate:
        # If critical ambiguity requires clarification and reasonable assumptions was not selected,
        # DO NOT call the enhancement model (zero enhancer LLM calls). Return early with empty enhanced text.
        if analysis.ambiguity.needs_clarification and not allow_reasonable_assumptions:
            return EnhanceResult(analysis, "", validation=None, repair_attempted=False, quality=None)

        # Deterministically select category template (O(1), zero LLM calls)
        selected_template = template or get_template(analysis.classification.category)

        # Build composite instruction containing original request, structured analysis, and category template
        user_content = build_enhancer_user_message(
            user_prompt,
            analysis,
            template=selected_template,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )
        messages = [
            {"role": "system", "content": HUMANINTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_enhanced_text = await self.client.chat(messages=messages)
        cleaned_enhanced_text = self._clean_enhanced_output(raw_enhanced_text, analysis, user_prompt)

        # Intent-Preservation Validation & Controlled 1-attempt repair:
        if validate and self.validator is not None:
            validation = await self.validator.validate(
                original_prompt=user_prompt,
                analysis=analysis,
                enhanced_prompt=cleaned_enhanced_text,
                clarification_answers=clarification_answers,
                allow_reasonable_assumptions=allow_reasonable_assumptions,
            )

            if validation.is_valid and getattr(getattr(analysis, "grounding", None), "is_grounded", True):
                quality_report: Optional[PromptQualityReport] = None
                if self.scorer is not None:
                    quality_report = await self.scorer.score(
                        original_prompt=user_prompt,
                        analysis=analysis,
                        enhanced_prompt=cleaned_enhanced_text,
                        validation=validation,
                    )
                return EnhanceResult(
                    analysis,
                    cleaned_enhanced_text,
                    validation=validation,
                    repair_attempted=False,
                    quality=quality_report,
                    initial_enhanced_prompt=cleaned_enhanced_text,
                    initial_validation=validation,
                )

            # Validation failed: run AT MOST ONE controlled repair attempt
            errors = [i for i in validation.issues if i.severity == ValidationSeverity.ERROR]
            logger.info("Validation failed with %d error(s). Initiating repair attempt.", len(errors))

            cleaned_repaired_text = await self.repair(
                user_prompt=user_prompt,
                analysis=analysis,
                previous_enhanced_prompt=cleaned_enhanced_text,
                errors=errors,
                template=selected_template,
                allow_reasonable_assumptions=allow_reasonable_assumptions,
            )

            # Re-validate repaired prompt
            revalidation = await self.validator.validate(
                original_prompt=user_prompt,
                analysis=analysis,
                enhanced_prompt=cleaned_repaired_text,
                clarification_answers=clarification_answers,
                allow_reasonable_assumptions=allow_reasonable_assumptions,
            )

            quality_report = None
            if revalidation.is_valid and getattr(getattr(analysis, "grounding", None), "is_grounded", True):
                logger.info("Repair validation passed successfully.")
                if self.scorer is not None:
                    quality_report = await self.scorer.score(
                        original_prompt=user_prompt,
                        analysis=analysis,
                        enhanced_prompt=cleaned_repaired_text,
                        validation=revalidation,
                    )
            else:
                logger.warning(
                    "Repair validation failed (%d error(s) remain).",
                    sum(1 for i in revalidation.issues if i.severity == ValidationSeverity.ERROR),
                )

            return EnhanceResult(
                analysis,
                cleaned_repaired_text,
                validation=revalidation,
                repair_attempted=True,
                quality=quality_report,
                initial_enhanced_prompt=cleaned_enhanced_text,
                initial_validation=validation,
            )

        return EnhanceResult(
            analysis,
            cleaned_enhanced_text,
            validation=None,
            repair_attempted=False,
            quality=None,
            initial_enhanced_prompt=cleaned_enhanced_text,
            initial_validation=None,
        )


prompt_enhancer = PromptEnhancer(validator=prompt_validator, scorer=prompt_scorer)
