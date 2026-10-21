"""Phase 6 General Intent-Preservation Validator."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import ValidationError

from app.core.intent_anchor import extract_intent_anchor
from app.core.instruction_boundary import scan_enhanced_prompt_contamination
from app.core.source_coverage import extract_source_semantic_units, is_semantic_unit_covered
from app.llm.ollama_client import OllamaResponseError, ollama_client
from app.prompts.validator import VALIDATOR_SYSTEM_PROMPT, build_validator_user_message
from app.schemas.analysis import ActionFamily, ClarificationAnswer, IntentAnchor, PromptAnalysis
from app.schemas.validation import (
    PromptValidation,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
)

logger = logging.getLogger("humanintent.validator")


class PromptValidator:
    """Validates that an enhanced prompt faithfully represents the user's intent.

    Uses a hybrid architecture:
    A. Deterministic universal invariant checks (fast, guaranteed invariants).
    B. LLM semantic verification with structured JSON schema output (deep semantic check).
    """

    def __init__(self, client=None) -> None:
        self.client = client or ollama_client

    def _build_schema_contract(self) -> Dict[str, Any]:
        """Generate JSON schema contract for PromptValidation."""
        schema = PromptValidation.model_json_schema()
        schema["required"] = ["is_valid", "issues"]
        return schema

    def _clean_json_content(self, raw_content: str) -> str:
        """Strip markdown fences or extraneous text around JSON if present."""
        trimmed = raw_content.strip()
        if trimmed.startswith("```"):
            trimmed = re.sub(r"^```(?:json)?\s*", "", trimmed, flags=re.IGNORECASE)
            trimmed = re.sub(r"\s*```$", "", trimmed)
        return trimmed.strip()

    def _parse_and_validate(
        self, raw_output: str
    ) -> Tuple[Optional[PromptValidation], Optional[str], Optional[Exception]]:
        """Clean, parse JSON, and validate against PromptValidation schema."""
        cleaned = self._clean_json_content(raw_output)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}", exc

        if not isinstance(parsed, dict):
            return None, "response is not a JSON object", ValueError("Response is not a JSON object")

        try:
            validation = PromptValidation.model_validate(parsed)
            return validation, None, None
        except ValidationError as exc:
            return None, f"does not match PromptValidation schema: {exc}", exc

    def _check_deterministic_invariants(
        self,
        original_prompt: str,
        analysis: PromptAnalysis,
        enhanced_prompt: str,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> list[ValidationIssue]:
        """Perform domain-agnostic deterministic invariant checks.

        Covers universal invariants:
        1. Non-empty enhancement.
        2. No self-referential HumanIntent meta-commentary.
        3. Explicit quantities / counts preserved and not contradicted.
        4. Negative constraints not contradicted.
        5. Rejected / superseded choices not reintroduced as active requirements.
        """
        issues: list[ValidationIssue] = []
        enhanced_stripped = enhanced_prompt.strip()

        # 1. Non-empty enhancement
        if not enhanced_stripped:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.OBJECTIVE_DRIFT,
                    severity=ValidationSeverity.ERROR,
                    message="Enhanced prompt is empty or contains only whitespace.",
                    evidence=None,
                )
            )
            return issues

        enhanced_lower = enhanced_stripped.lower()
        original_lower = original_prompt.lower()

        # 2. Self-referential meta-commentary & internal instruction/control leakage
        contamination_issues = scan_enhanced_prompt_contamination(
            enhanced_stripped, original_prompt, clarification_answers
        )
        if contamination_issues:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.META_COMMENTARY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Enhanced prompt contains internal system instructions or validation meta-commentary: {contamination_issues[0]}",
                    evidence=contamination_issues[0],
                )
            )
        else:
            meta_matches = re.findall(
                r"\b(?:HumanIntent|intent-preserving prompt optimizer|Here is the enhanced prompt:?)\b",
                enhanced_stripped,
                re.IGNORECASE,
            )
            if meta_matches:
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.META_COMMENTARY,
                        severity=ValidationSeverity.ERROR,
                        message="Enhanced prompt contains self-referential system commentary.",
                        evidence=meta_matches[0],
                    )
                )

        # 3. Explicit numeric quantities / limits
        # Look for patterns like "5 ideas", "3 paragraphs", "under 150 words", "10 questions"
        quantity_patterns = [
            r"\b(\d+)\s+([a-zA-Z]{3,})\b",
            r"\b(?:under|maximum|max|at most|up to|limit\s+to)\s+(\d+)\s+([a-zA-Z]{3,})\b",
        ]
        quantity_patterns = [
            r"\b(\d+)\s+([a-zA-Z]{3,})\b",
            r"\b(?:under|maximum|max|at most|up to|limit\s+to)\s+(\d+)\s+([a-zA-Z]{3,})\b",
        ]
        requested_quantities: dict[str, set[str]] = {}
        for pattern in quantity_patterns:
            for match in re.finditer(pattern, original_lower):
                num, noun = match.groups()
                # Exclude years or generic numbers
                if int(num) > 1900 and int(num) < 2100:
                    continue
                noun_singular = noun.rstrip("s")
                requested_quantities.setdefault(noun_singular, set()).add(num)

        # Also check quantities in explicit_requirements and constraints
        for req in analysis.explicit_requirements:
            for pattern in quantity_patterns:
                for match in re.finditer(pattern, req.lower()):
                    num, noun = match.groups()
                    if int(num) > 1900 and int(num) < 2100:
                        continue
                    noun_singular = noun.rstrip("s")
                    requested_quantities.setdefault(noun_singular, set()).add(num)

        for const in analysis.constraints:
            for pattern in quantity_patterns:
                for match in re.finditer(pattern, const.lower()):
                    num, noun = match.groups()
                    if int(num) > 1900 and int(num) < 2100:
                        continue
                    noun_singular = noun.rstrip("s")
                    requested_quantities.setdefault(noun_singular, set()).add(num)

        for noun_singular, allowed_nums in requested_quantities.items():
            # Check if the enhanced prompt has an unrequested number for this noun
            enhanced_noun_matches = re.findall(
                rf"\b(\d+)\s+{re.escape(noun_singular)}s?\b", enhanced_lower
            )
            if enhanced_noun_matches:
                for enh_num in enhanced_noun_matches:
                    if enh_num not in allowed_nums:
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.QUANTITY_MISMATCH,
                                severity=ValidationSeverity.ERROR,
                                message=f"Explicit quantity mismatch for '{noun_singular}': requested {allowed_nums}, but enhanced prompt specifies {enh_num}.",
                                evidence=f"Requested: {allowed_nums} {noun_singular}s | Found: {enh_num} {noun_singular}s",
                            )
                        )
                        break

        # 4. Negative constraints contradiction
        # Extract negative constraints from analysis.constraints and prompt
        IDIOMATIC_WORDS = {
            "exception", "fail", "delay", "doubt", "hesitation", "issue", "issues",
            "limitation", "restrictions", "restriction", "interruption", "downtime",
            "loss", "prejudice", "regard", "any", "a", "an", "the", "it", "problem",
            "other", "more", "less",
        }
        negative_patterns = [
            r"\b(?:do\s+not\s+use|don't\s+use|avoid|without|never\s+use|no)\s+([a-zA-Z0-9_\-\.\+]+)\b",
        ]
        forbidden_items: set[str] = set()
        for constraint in analysis.constraints:
            c_lower = constraint.lower()
            for pat in negative_patterns:
                for match in re.finditer(pat, c_lower):
                    item = match.group(1).strip()
                    if len(item) > 2 and item not in IDIOMATIC_WORDS:
                        forbidden_items.add(item)

        for pat in negative_patterns:
            for match in re.finditer(pat, original_lower):
                item = match.group(1).strip()
                if len(item) > 2 and item not in IDIOMATIC_WORDS:
                    forbidden_items.add(item)

        for item in forbidden_items:
            # Check if enhanced prompt directs using the forbidden item
            pos_usage_patterns = [
                rf"\b(?:use|using|utilize|utilizing|implement\s+with|require|include)\s+{re.escape(item)}\b",
                rf"\b(?:written\s+in|built\s+with)\s+{re.escape(item)}\b",
            ]
            for pos_pat in pos_usage_patterns:
                if re.search(pos_pat, enhanced_lower):
                    # Ensure it's not negated in the enhanced prompt
                    neg_guard = rf"\b(?:do\s+not|don't|no|never|avoid|without)\s+(?:use\s+|using\s+)?{re.escape(item)}\b"
                    if not re.search(neg_guard, enhanced_lower):
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.CONSTRAINT_CONTRADICTION,
                                severity=ValidationSeverity.ERROR,
                                message=f"Enhanced prompt prescribes '{item}', which was explicitly prohibited by user constraint.",
                                evidence=f"Prohibited: {item}",
                            )
                        )
                        break

        # 5. Rejected / Superseded options reintroduced
        # Check for user self-corrections like "not X, rather Y", "wait maybe like a calculator? no like a quiz"
        rejected_patterns = [
            r"\b([a-zA-Z0-9_\-\.]+)\b\s*[\?\!\,\.\;\—\-]*\s*(?:wait\s+no|actually\s+no|actually\s+use|scratch\s+that|no\s+like|change\s+that\s+to)",
            r"\b(?:not|no|rather\s+than|instead\s+of)\s+(?:a\s+|an\s+)?([a-zA-Z0-9_\-\.]+)\b",
        ]
        rejected_terms: set[str] = set()
        for pat in rejected_patterns:
            for match in re.finditer(pat, original_lower):
                term = match.group(1).strip()
                if len(term) > 2 and term not in {"the", "a", "an", "this", "that", "it", "normal", "simple"}:
                    rejected_terms.add(term)

        for term in rejected_terms:
            # If the term appears in enhanced prompt as a deliverable/active requirement
            # e.g., "Build a calculator", "Create a calculator", "Develop a calculator"
            term_as_active = [
                rf"\b(?:build|create|develop|implement|design|provide)\s+(?:a\s+|an\s+)?{re.escape(term)}\b",
                rf"\b(?:primary\s+objective|deliverable|goal)[\s\:\-]+[^\.\n]*?\b{re.escape(term)}\b",
            ]
            if any(re.search(p, enhanced_lower) for p in term_as_active):
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.REJECTED_OPTION_REINTRODUCED,
                        severity=ValidationSeverity.ERROR,
                        message=f"Enhanced prompt reintroduces '{term}', which was superseded or rejected by the user.",
                        evidence=f"Rejected option: {term}",
                    )
                )

        # 6. Source-of-truth grounding validation
        # Ground truth is authoritative user text: original_prompt + clarification_answers
        user_corpus = original_lower
        if clarification_answers:
            user_corpus += " " + " ".join(ca.answer.lower() for ca in clarification_answers if ca.answer)

        # 6a. Direct authoritative source-to-enhancement requirement verification
        source_units = (
            analysis.grounding.source_units
            if (hasattr(analysis, "grounding") and analysis.grounding and analysis.grounding.source_units)
            else extract_source_semantic_units(original_prompt, clarification_answers)
        )
        if len(source_units) >= 2:
            for unit in source_units:
                if not is_semantic_unit_covered(unit, enhanced_lower):
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.SOURCE_REQUIREMENT_DROPPED,
                            severity=ValidationSeverity.ERROR,
                            message=f"Explicit source requirement '{unit}' was omitted from enhanced prompt.",
                            evidence=f"Authoritative source requirement: {unit}",
                        )
                    )

        # 6b. Check if analysis itself has ungrounded assertions flag
        if hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.ANALYSIS_GROUNDING_FAILURE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Structured analysis contains ungrounded assertions or omitted source requirements: {', '.join(analysis.grounding.unsupported_items)}",
                    evidence=str(analysis.grounding.unsupported_items),
                )
            )

        # 6b. Check for mutual hallucination / objective drift where analysis and enhancement agree on B, but user asked A
        stop_words = {
            "the", "a", "an", "for", "to", "in", "on", "at", "by", "with", "from",
            "and", "or", "is", "it", "my", "me", "we", "our", "you", "your", "that",
            "this", "these", "those", "can", "could", "would", "should", "want", "need",
            "please", "help", "do", "does", "did", "have", "has", "had", "be", "been",
            "being", "so", "like", "wait", "maybe", "actually", "just", "make", "create",
            "build", "write", "give", "task", "prompt", "using", "into", "about",
        }
        user_domain_words = {
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", user_corpus)
            if w not in stop_words
        }

        obj_lower = (analysis.objective or "").lower()
        # If user provided concrete domain terms (>= 2 terms) but NONE appear in objective or enhanced prompt
        if len(user_domain_words) >= 2:
            matched_in_enh = sum(1 for w in user_domain_words if w in enhanced_lower or w in obj_lower)
            if matched_in_enh == 0:
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.ANALYSIS_OBJECTIVE_DRIFT,
                        severity=ValidationSeverity.ERROR,
                        message="Enhanced prompt and analysis implement a task completely drifted from the original user request.",
                        evidence=f"User requested topics: {sorted(list(user_domain_words))[:5]} | Analysis objective: '{analysis.objective}'",
                    )
                )

        # 6c. Check for ungrounded explicit technologies asserted in analysis and adopted in enhanced prompt
        known_frameworks = {
            "react", "vue", "angular", "svelte", "django", "flask", "fastapi",
            "express", "node", "nodejs", "sqlite", "postgres", "postgresql",
            "mongodb", "mysql", "bootstrap", "tailwind", "firebase", "aws",
        }
        for req in analysis.explicit_requirements:
            req_lower = req.strip().lower()
            if req_lower in known_frameworks and req_lower not in user_corpus:
                if req_lower in enhanced_lower:
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.UNSUPPORTED_ANALYSIS_REQUIREMENT,
                            severity=ValidationSeverity.ERROR,
                            message=f"Explicit requirement '{req}' was asserted and adopted without user grounding.",
                            evidence=f"Unprompted technology: {req}",
                        )
                    )

        # 6d. Check for semantic grounding mismatch (modifier modifying artifact turned into explanation/tone preference)
        for pref in analysis.preferences:
            pref_l = pref.lower()
            if any(term in pref_l for term in ["simple explanation", "explain simply", "simple language"]):
                if re.search(r"\bsimple\s+(?:web\s+app|app|application|tool|script|dashboard|page|calculator|system|database|api|program|bot|workflow|model|service)\b", user_corpus):
                    if not re.search(r"\b(?:explain|explanation|terms|words|concept|breakdown)\b", user_corpus):
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.SEMANTIC_GROUNDING_MISMATCH,
                                severity=ValidationSeverity.ERROR,
                                message=f"Preference '{pref}' misinterprets artifact scope modifier 'simple' as an explanation style.",
                                evidence=f"Artifact modifier 'simple' context: '{pref}'",
                            )
                        )
            if "professional tone" in pref_l or "sound professional" in pref_l:
                if re.search(r"\bprofessional\s+(?:portfolio|network|networking|headshot|photo|camera|equipment|software|tool|app|website)\b", user_corpus):
                    if not re.search(r"\b(?:tone|manner|email|letter|message|memo|writing|write|draft)\b", user_corpus):
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.SEMANTIC_GROUNDING_MISMATCH,
                                severity=ValidationSeverity.ERROR,
                                message=f"Preference '{pref}' misinterprets artifact attribute 'professional' as writing tone.",
                                evidence=f"Artifact attribute 'professional' context: '{pref}'",
                            )
                        )
            if "formal tone" in pref_l:
                if re.search(r"\bformal\s+(?:verification|logic|methods?|proof|grammar|system|semantics)\b", user_corpus):
                    if not re.search(r"\b(?:tone|manner|email|letter|message|writing|write)\b", user_corpus):
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.SEMANTIC_GROUNDING_MISMATCH,
                                severity=ValidationSeverity.ERROR,
                                message=f"Preference '{pref}' misinterprets domain method modifier 'formal' as writing tone.",
                                evidence=f"Domain modifier 'formal' context: '{pref}'",
                            )
                        )

        # 6e. Check for invented functional purpose when artifact purpose was unresolved
        if hasattr(analysis, "deliverable_semantics") and analysis.deliverable_semantics:
            if not analysis.deliverable_semantics.is_functionally_complete and not allow_reasonable_assumptions:
                invented_domain_patterns = [
                    r"\b(?:customer\s+support|e-commerce|ecommerce|crypto|cryptocurrency|flight\s+booking|hotel\s+reservation|medical\s+records|real\s+estate|weather\s+forecast)\b",
                ]
                for idp in invented_domain_patterns:
                    m = re.search(idp, enhanced_lower)
                    if m and not re.search(idp, user_corpus):
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.INVENTED_REQUIREMENT,
                                severity=ValidationSeverity.ERROR,
                                message=f"Enhanced prompt invents a specific functional purpose '{m.group(0)}' for an ungrounded artifact container.",
                                evidence=f"Invented functional domain: {m.group(0)}",
                            )
                        )

        # 6f. Check Task Family Drift against authoritative IntentAnchor
        anchor = getattr(analysis, "intent_anchor", None) or extract_intent_anchor(original_prompt, clarification_answers)
        if anchor and anchor.is_strong:
            if anchor.action_family == ActionFamily.SOFTWARE_CREATION:
                if (
                    re.search(r"\b(?:draft\s+(?:a\s+|an\s+)?(?:email|letter|memo|speech)|write\s+(?:a\s+|an\s+)?(?:poem|story|novel|essay|press\s+release))\b", enhanced_lower)
                    and not any(w in enhanced_lower for w in ["code", "function", "script", "software", "program", "app", "application", "tool"])
                ):
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.TASK_FAMILY_DRIFT,
                            severity=ValidationSeverity.ERROR,
                            message="Enhanced prompt drifted from software creation task family: converted software implementation to prose drafting.",
                            evidence=f"Action family: {anchor.action_family.value} ({anchor.primary_action})",
                        )
                    )
            elif anchor.action_family == ActionFamily.WRITING:
                if (
                    re.search(r"\b(?:build|create|implement|code)\s+(?:an?\s+)?(?:[a-z\-]+\s+)?(?:web\s+app|app|application|portal|service|database|api|script|tool)\b", enhanced_lower)
                    or "database schema" in enhanced_lower
                ) and not any(w in original_lower for w in ["app", "software", "code", "database", "api", "tool"]):
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.TASK_FAMILY_DRIFT,
                            severity=ValidationSeverity.ERROR,
                            message="Enhanced prompt drifted from writing task family: converted text drafting task to software implementation.",
                            evidence=f"Action family: {anchor.action_family.value} ({anchor.primary_action})",
                        )
                    )
            elif anchor.action_family == ActionFamily.EDUCATION_EXPLANATION:
                if any(w in enhanced_lower for w in ["build an app", "create an application", "deploy a service"]) and not any(w in original_lower for w in ["app", "build", "deploy"]):
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.TASK_FAMILY_DRIFT,
                            severity=ValidationSeverity.ERROR,
                            message="Enhanced prompt drifted from educational explanation task family: converted explanation to software creation.",
                            evidence=f"Action family: {anchor.action_family.value} ({anchor.primary_action})",
                        )
                    )
            elif anchor.action_family == ActionFamily.PLANNING:
                if any(w in enhanced_lower for w in ["build an app", "write a python script", "implement a function"]) and not any(w in original_lower for w in ["app", "script", "function", "code"]):
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.TASK_FAMILY_DRIFT,
                            severity=ValidationSeverity.ERROR,
                            message="Enhanced prompt drifted from planning task family: converted planning/schedule to software creation.",
                            evidence=f"Action family: {anchor.action_family.value} ({anchor.primary_action})",
                        )
                    )

        return issues

    async def validate(
        self,
        original_prompt: str,
        analysis: PromptAnalysis,
        enhanced_prompt: str,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> PromptValidation:
        """Validate an enhanced prompt against user intent using deterministic invariants and LLM verification."""
        # Step 1: Run deterministic invariant checks
        deterministic_issues = self._check_deterministic_invariants(
            original_prompt=original_prompt,
            analysis=analysis,
            enhanced_prompt=enhanced_prompt,
            clarification_answers=clarification_answers,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )

        # If deterministic checks already found an empty prompt or meta-commentary,
        # we still run or can return early if empty
        if not enhanced_prompt.strip():
            logger.warning("Validation failed: enhanced prompt is empty.")
            return PromptValidation(is_valid=False, issues=deterministic_issues)

        # Step 2: Run LLM semantic validation
        user_message = build_validator_user_message(
            original_prompt=original_prompt,
            analysis=analysis,
            enhanced_prompt=enhanced_prompt,
            clarification_answers=clarification_answers,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )

        schema = self._build_schema_contract()
        messages = [
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # ATTEMPT 1: Primary validation call using native JSON schema
        raw_output = await self.client.chat(messages=messages, format=schema, num_predict=600)
        llm_validation, issue, exc = self._parse_and_validate(raw_output)

        # If Attempt 1 failed, trigger controlled recovery (MAXIMUM 1 RETRY)
        if llm_validation is None:
            logger.warning(
                "Validator attempt 1 failed schema validation: %s. Initiating recovery attempt.",
                issue,
            )
            repair_messages = [
                {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Previous validation output failed schema validation ({issue}).\n"
                        f"Evaluate the prompt again and return a strict PromptValidation JSON object:\n"
                        f"- is_valid: boolean (False if any ERROR exists, True otherwise)\n"
                        f"- issues: list of objects with 'type', 'severity' (INFO, WARNING, or ERROR), 'message', 'evidence'\n"
                        f"Original task input:\n{user_message}"
                    ),
                },
            ]
            raw_repair_output = await self.client.chat(messages=repair_messages, format=schema, num_predict=600)
            llm_validation, repair_issue, repair_exc = self._parse_and_validate(raw_repair_output)

            if llm_validation is None:
                logger.error(
                    "Validator recovery attempt also failed schema validation: %s. Falling back to deterministic results.",
                    repair_issue,
                )
                has_det_err = any(i.severity == ValidationSeverity.ERROR for i in deterministic_issues)
                return PromptValidation(is_valid=not has_det_err, issues=deterministic_issues)

        # Step 3: Combine issues from deterministic checks and LLM validation
        combined_issues: list[ValidationIssue] = list(deterministic_issues)
        seen_keys: set[str] = {
            f"{i.type.value}:{i.message.lower()[:30]}" for i in deterministic_issues
        }

        for issue in llm_validation.issues:
            # Filter spurious issues that flag intrinsic technical medium as an unrequested technology or business domain
            msg_lower = (issue.message or "").lower()
            ev_lower = (issue.evidence or "").lower()
            combined_text = f"{msg_lower} {ev_lower}"
            if any(term in combined_text for term in ["'software development'", "software development", "'programming'", "'scripting'", "'coding'"]):
                if any(w in combined_text for w in ["unrequested technology", "business domain", "domain invention"]):
                    logger.info("Filtered spurious validator issue on generic medium: %s", issue.message)
                    continue

            key = f"{issue.type.value}:{issue.message.lower()[:30]}"
            if key not in seen_keys:
                combined_issues.append(issue)
                seen_keys.add(key)

        # If any ERROR exists, is_valid MUST be False
        has_error = any(i.severity == ValidationSeverity.ERROR for i in combined_issues)
        is_valid = not has_error

        error_count = sum(1 for i in combined_issues if i.severity == ValidationSeverity.ERROR)
        if is_valid:
            logger.info("Validation passed successfully (0 errors, %d warnings/infos).", len(combined_issues))
        else:
            logger.warning("Validation failed with %d error(s).", error_count)

        return PromptValidation(is_valid=is_valid, issues=combined_issues)


prompt_validator = PromptValidator()
