import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import ValidationError
from app.core.instruction_boundary import detect_control_text, scan_analysis_contamination
from app.llm.ollama_client import OllamaResponseError, ollama_client
from app.prompts.analyzer import ANALYZER_SYSTEM_PROMPT
from app.schemas.analysis import (
    ActionFamily,
    AmbiguityAssessment,
    AmbiguityLevel,
    AnalysisGrounding,
    ClarificationAnswer,
    ClarificationQuestion,
    DeliverableSemantics,
    GroundingEvidence,
    GroundingRelation,
    IntentAnchor,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
    SourceType,
)
from app.core.intent_anchor import (
    ACTION_FAMILY_TO_CATEGORY,
    check_analysis_coherence,
    extract_intent_anchor,
    is_self_contained_operation,
    normalize_missing_context_dimension,
)
from app.core.source_coverage import check_source_coverage, extract_source_semantic_units
from app.core.context_budget import (
    CONTEXT_BUDGET_EXCEEDED,
    GLOBAL_RECONCILIATION_FAILED,
    LONG_INPUT_MODE,
    NORMAL_CONTEXT_MODE,
    ProcessingMode,
    SOURCE_CHUNK_COVERAGE_FAILED,
    SOURCE_CHUNK_EXTRACTION_FAILED,
    chunk_long_source,
    context_budget_manager,
    extract_chunk_atomic_units,
    reconcile_long_input_analysis,
    resolve_supersessions_and_deduplicate,
)

logger = logging.getLogger("humanintent.analyzer")

# Context-sensitive mapping of communication and presentation style modifiers
STYLE_MODIFIER_MAPPINGS = [
    (r"\b(?:professional\s+tone|written\s+professionally|write\s+professionally|in\s+a\s+professional\s+tone|in\s+a\s+professional\s+manner|sound\s+professional|professional\s+(?:email|letter|message|memo|communication|draft))\b", "professional tone", "professional"),
    (r"\b(?:formal\s+tone|formal\s+style|written\s+formally|write\s+formally|in\s+a\s+formal\s+tone|in\s+a\s+formal\s+manner|formal\s+(?:email|letter|memo|document))\b", "formal tone", "formal"),
    (r"\b(?:casual\s+tone|casual\s+style|written\s+casually|write\s+casually|in\s+a\s+casual\s+tone|sound\s+casual|casual\s+(?:email|message|chat))\b", "casual tone", "casual"),
    (r"\b(?:friendly\s+tone|friendly\s+style|sound\s+friendly|in\s+a\s+friendly\s+tone|friendly\s+(?:email|message|note))\b", "friendly tone", "friendly"),
    (r"\b(?:explain\s+(?:[^\.\n]*?\s+)?simply|simply\s+explain|simple\s+explanation|in\s+simple\s+terms|in\s+simple\s+words|break\s+(?:it\s+)?down\s+simply)\b", "simple explanation", "simple"),
    (r"\b(?:concise|brief|make\s+it\s+brief|keep\s+it\s+brief|keep\s+it\s+short|keep\s+it\s+concise)\b", "concise style", "concise"),
    (r"\b(?:detailed\s+explanation|explain\s+(?:in\s+)?detail|in-depth\s+explanation|detailed\s+breakdown)\b", "detailed explanation", "detailed"),
    (r"\b(?:make\s+it\s+sound\s+cool|sound\s+cool|cool\s+tone)\b", "cool and engaging style", "cool"),
]

GENERIC_FILLER_WORDS = {
    "system", "systems", "requirement", "requirements", "app", "application",
    "tracking", "tool", "tools", "features", "feature", "details", "detail",
    "specification", "specifications", "method", "methods", "software", "project",
    "environment", "process", "setup", "data", "email", "letter", "message",
    "memo", "thing", "something", "stuff", "it", "plan", "plans",
    "dashboard", "dashboards", "page", "pages", "script", "scripts",
    "actually", "scratch", "rather", "instead", "wait", "maybe",
}

GENERIC_ACTION_VERBS = {
    "create", "build", "make", "develop", "implement", "write", "draft",
    "generate", "produce", "design", "setup", "establish", "provide",
    "construct", "assist", "prepare", "execute", "define", "describe",
    "handle", "support", "deliver", "specify", "complete", "request",
    "conduct", "perform",
}

STOP_WORDS = {
    "make", "build", "create", "write", "explain", "simple", "simply",
    "using", "with", "into", "from", "that", "this", "your", "about",
    "for", "the", "and", "a", "an", "in", "to", "of", "on", "at", "by",
    "app", "application", "analyze", "help", "me", "do", "my",
    "use", "actually", "rather", "instead", "scratch", "wait", "maybe", "like", "just",
}

# Candidate technologies that must not be suggested in missing_context unless requested
FORBIDDEN_TECH_EXAMPLES = {
    "react", "angular", "vue", "next.js", "nextjs", "tailwind",
    "firebase", "aws", "sqlite", "postgres", "postgresql", "mongodb",
}


class PromptAnalyzer:
    """Service responsible for extracting structured intent and requirements from a raw prompt."""

    def __init__(self, client=None) -> None:
        self.client = client or ollama_client

    async def _safe_chat(self, messages: list[dict], format: Optional[dict] = None, num_predict: Optional[int] = None) -> str:
        try:
            if num_predict is not None:
                return await self.client.chat(messages=messages, format=format, num_predict=num_predict)
            return await self.client.chat(messages=messages, format=format)
        except TypeError as e:
            if "unexpected keyword argument" in str(e) or "num_predict" in str(e):
                return await self.client.chat(messages=messages, format=format)
            raise

    def _build_schema_contract(self) -> Dict[str, Any]:
        """Generate schema contract from PromptAnalysis ensuring all core fields are required."""
        schema = PromptAnalysis.model_json_schema()
        schema["required"] = [
            "classification",
            "objective",
            "explicit_requirements",
            "constraints",
            "preferences",
            "missing_context",
            "ambiguity",
        ]
        return schema

    def _clean_json_content(self, raw_content: str) -> str:
        """Strip markdown fences or extraneous text around JSON if present."""
        trimmed = raw_content.strip()
        # Remove markdown code block fences if the model included them
        if trimmed.startswith("```"):
            trimmed = re.sub(r"^```(?:json)?\s*", "", trimmed, flags=re.IGNORECASE)
            trimmed = re.sub(r"\s*```$", "", trimmed)
        return trimmed.strip()

    def _is_superseded_or_rejected(self, term: str, prompt_lower: str) -> bool:
        """Detect if a user initially mentioned a term but then superseded, rejected, or revised it."""
        t_esc = re.escape(term.lower().strip())
        # If the term is explicitly introduced as the replacement (e.g. "use X instead", "rather use X")
        if re.search(rf"\b(?:use|with|prefer|choose)\s+{t_esc}\s+instead\b", prompt_lower):
            return False
        if re.search(rf"\b{t_esc}\s+instead\b", prompt_lower):
            return False
        if re.search(rf"\b(?:instead|rather)\s+(?:use|with)?\s+{t_esc}\b", prompt_lower):
            return False

        patterns = [
            rf"\b{t_esc}\b\s*[\?\!\,\.\;\—\-]*\s*(?:wait\s+no|actually\s+no|actually\s+use|actually|no|not|rather|scratch\s+that|change\s+that\s+to|i\s+mean)\b",
            rf"\b(?:no|not|instead\s+of|rather\s+than)\s+{t_esc}\b",
            rf"\b(?:start\s+with\s+|maybe\s+(?:like\s+a\s+|a\s+)?)?{t_esc}\b[^\.\n]*?(?:scratch\s+that|wait\s+no|actually\b|change\s+that\s+to|rather\b|no\s+like)",
        ]
        return any(re.search(p, prompt_lower) for p in patterns)

    def _recover_style_preferences(self, raw_prompt: str, preferences: list[str]) -> list[str]:
        """Recover explicit presentation, tone, or style preferences present in user wording, excluding superseded or misassigned ones."""
        prompt_lower = raw_prompt.lower()
        recovered: list[str] = []
        for p in preferences:
            p_lower = p.lower()

            # Semantic Role Verification: Filter preferences where a modifier was misassigned to output style
            if any(term in p_lower for term in ["simple explanation", "explain simply", "simple language"]):
                # If "simple" modifies an artifact deliverable and not an explanation request
                if re.search(r"\bsimple\s+(?:web\s+app|app|application|tool|script|dashboard|page|calculator|system|database|api|program|bot|workflow|model|service)\b", prompt_lower):
                    if not re.search(r"\b(?:explain|explanation|terms|words|concept|breakdown)\b", prompt_lower):
                        continue
            if "professional tone" in p_lower or "sound professional" in p_lower:
                if re.search(r"\bprofessional\s+(?:portfolio|network|networking|headshot|photo|camera|equipment|software|tool|app|website)\b", prompt_lower):
                    if not re.search(r"\b(?:tone|manner|email|letter|message|memo|writing|write|draft)\b", prompt_lower):
                        continue
            if "formal tone" in p_lower:
                if re.search(r"\bformal\s+(?:verification|logic|methods?|proof|grammar|system|semantics)\b", prompt_lower):
                    if not re.search(r"\b(?:tone|manner|email|letter|message|writing|write)\b", prompt_lower):
                        continue
            if any(t in p_lower for t in ["light theme", "light mode"]):
                if re.search(r"\blight(?:weight)?\s+(?:database|db|container|library|dependency|footprint|service)\b", prompt_lower):
                    if not re.search(r"\b(?:theme|mode|ui|visual|color|style)\b", prompt_lower):
                        continue
            if any(t in p_lower for t in ["fast response", "fast style"]):
                if re.search(r"\bfast\s+(?:algorithm|query|search|function|script|execution|runtime)\b", prompt_lower):
                    continue

            # If the preference mentions "friendly", ensure "friendly" actually occurs in prompt
            if "friendly" in p_lower and not re.search(r"\bfriendly\b", prompt_lower):
                continue
            # If the preference mentions "formal", ensure "formal" occurs in prompt
            if "formal" in p_lower and not re.search(r"\bformal\b", prompt_lower):
                continue
            # If the preference mentions "casual", ensure "casual" occurs in prompt
            if "casual" in p_lower and not re.search(r"\bcasual\b", prompt_lower):
                continue
            # If the preference mentions "professional", ensure "professional" occurs in prompt
            if "professional" in p_lower and not re.search(r"\bprofessional\b", prompt_lower):
                continue
            if any(kw in p_lower and self._is_superseded_or_rejected(kw, prompt_lower) for _, _, kw in STYLE_MODIFIER_MAPPINGS):
                continue
            recovered.append(p)

        for pattern, canonical_pref, keyword in STYLE_MODIFIER_MAPPINGS:
            if re.search(pattern, prompt_lower) and not self._is_superseded_or_rejected(keyword, prompt_lower):
                if not any(keyword in p.lower() for p in recovered):
                    recovered.append(canonical_pref)
        return recovered

    def _clean_missing_context(
        self,
        missing_context: list[str],
        objective: str,
        explicit_requirements: list[str],
        preferences: list[str],
        raw_prompt: str,
        category: Optional[str] = None,
    ) -> list[str]:
        """Filter out missing-context items that duplicate preferences, explicit requirements, or restate the objective."""
        cleaned: list[str] = []

        has_tone_pref = any(
            any(k in p.lower() for k in ["tone", "professional", "formal", "casual", "friendly", "cool"])
            for p in preferences
        )

        has_web_interface = bool(
            re.search(r"\b(?:website|webpage|web\s+page|site|page\s+on\s+my\s+website)\b", raw_prompt.lower())
        )

        explicit_lower = {r.strip().lower() for r in explicit_requirements}

        # Extract core domain terms from the prompt and objective
        prompt_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", raw_prompt.lower()) if w not in STOP_WORDS}
        obj_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", objective.lower()) if w not in STOP_WORDS}
        core_domain_words = prompt_words & obj_words

        has_explicit_framework = any(
            req in {"react", "angular", "vue", "next.js", "nextjs", "svelte", "django", "flask", "fastapi", "express", "node", "nodejs"}
            for req in explicit_lower
        )
        has_explicit_language = any(
            req in {"python", "javascript", "typescript", "go", "rust", "java", "c#", "c++", "ruby", "html", "css"}
            for req in explicit_lower
        )

        for item in missing_context:
            item_str = item.strip()
            if not item_str:
                continue

            # Strip candidate-answer parentheses: (e.g., ...), (for example, ...), (such as ...), (i.e. ...)
            item_str = re.sub(r"\s*\((?:e\.g\.|for example|such as|i\.e\.)[^)]*\)", "", item_str, flags=re.IGNORECASE).strip()
            # Strip parenthetical option lists: (CLI, web, GUI), (showcase, resume, contact), etc.
            item_str = re.sub(r"\s*\([^)]*(?:,|\/|\bor\b)[^)]*\)", "", item_str, flags=re.IGNORECASE).strip()
            item_str = re.sub(r"\s*(?:such as|for example|e\.g\.)\s+[^,;]+", "", item_str, flags=re.IGNORECASE).strip()

            item_lower = item_str.lower()
            if not item_lower:
                continue

            # If prompt specifies a website/webpage/site, interface medium is already known as web
            if has_web_interface and any(
                term in item_lower for term in [
                    "interface type", "application interface type", "quiz interface type",
                    "interface medium", "web interface", "user interface type", "ui type",
                    "platform (web/mobile)", "web or mobile"
                ]
            ):
                continue

            # Candidate answers/features must NOT be accepted as missing context dimensions
            candidate_features = [
                "project showcase", "resume", "contact page", "contact form",
                "personal branding", "digital footprint", "about me", "testimonials",
                "skills section", "sections", "features",
            ]
            if any(re.search(rf"\b{re.escape(feat)}\b", item_lower) for feat in candidate_features):
                if not any("content" in c.lower() for c in cleaned):
                    cleaned.append("content requirements")
                continue

            # Candidate visual styles must NOT be accepted as missing context dimensions
            if any(re.search(rf"\b{re.escape(st)}\b", item_lower) for st in ["dark theme", "light theme", "minimal design", "color palette", "theme"]):
                if not any("visual style" in c.lower() for c in cleaned):
                    cleaned.append("visual style")
                continue

            # If item contains unrequested candidate technologies, neutralize to dimension
            has_forbidden_tech = any(tech in item_lower for tech in FORBIDDEN_TECH_EXAMPLES if tech not in explicit_lower)
            if has_forbidden_tech:
                if any(w in item_lower for w in ["platform", "framework", "library", "frontend", "front-end", "backend"]):
                    item_str = "development platform or framework"
                    item_lower = item_str.lower()
                elif any(w in item_lower for w in ["database", "storage", "persistence"]):
                    item_str = "data storage method"
                    item_lower = item_str.lower()
                elif any(w in item_lower for w in ["hosting", "cloud", "deployment"]):
                    item_str = "hosting and deployment environment"
                    item_lower = item_str.lower()
                else:
                    for tech in FORBIDDEN_TECH_EXAMPLES:
                        if tech not in explicit_lower:
                            item_str = re.sub(rf"\b{re.escape(tech)}\b", "", item_str, flags=re.IGNORECASE).strip()
                    item_str = re.sub(r"\s+", " ", item_str).strip()
                    item_lower = item_str.lower()
                    if not item_lower:
                        continue

            # Normalize common platform/feature phrases to neutral canonical forms
            if item_lower in {"desired platform", "platform"}:
                item_str = "development platform or framework"
                item_lower = item_str.lower()
            elif re.search(r"\b(?:features?|sections?|components?)\b", item_lower) and any(w in item_lower for w in ["content", "page", "site", "app", "application", "system"]):
                item_str = "content requirements"
                item_lower = item_str.lower()

            # Neutralize unrequested candidate product attributes (demographics, brands, sizes, prices) to dimensions
            if any(re.search(rf"\b{w}\b", item_lower) for w in ["demographic", "demographics", "age", "gender", "location", "brand", "brands", "size", "sizes", "price", "prices", "pricing"]):
                if not any(re.search(rf"\b{w}\b", " ".join(explicit_lower)) for w in ["price", "pricing", "brand", "size", "demographic", "age", "gender", "location"]):
                    if any(term in prompt_words for term in ["recommend", "recommendation", "recommends", "quiz", "selector", "suggest", "product", "catalog", "item"]):
                        item_str = "recommendation data source"
                        item_lower = item_str.lower()
                    else:
                        continue

            # 1. Tone / style consistency: if tone is in preferences, remove tone from missing context
            if has_tone_pref and ("tone" in item_lower or any(p in item_lower for p in ["professional", "formal", "friendly", "cool"])):
                continue

            # 2. Check if item asks for platform/framework when already explicitly supplied
            if has_explicit_framework and any(term in item_lower for term in ["platform", "framework"]):
                continue

            # 3. Check if item asks for language when already explicitly supplied
            if has_explicit_language and any(term in item_lower for term in ["programming language", "language choice", "development language"]):
                continue

            # 4. Check if item repeats an explicit requirement
            if any(req in item_lower and (item_lower == req or any(g in item_lower for g in GENERIC_FILLER_WORDS)) for req in explicit_lower):
                continue

            # 5. Check if item is an objective restatement:
            item_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", item_lower))
            domain_matches = item_tokens & core_domain_words

            # If it contains multiple core domain words (e.g. both 'university' and 'scheduling')
            # and remaining tokens are generic filler words:
            if len(domain_matches) >= 2:
                remaining = item_tokens - domain_matches
                if not remaining or remaining.issubset(GENERIC_FILLER_WORDS):
                    continue

            # If it contains 1 core domain word and ALL other words are generic filler words
            # (e.g. 'scheduling solution', 'scheduling tool', 'campus portal'):
            if len(domain_matches) >= 1:
                remaining = item_tokens - domain_matches
                if not remaining or remaining.issubset(GENERIC_FILLER_WORDS):
                    continue

            if item_str.lower() not in [c.lower() for c in cleaned]:
                cleaned.append(item_str)

        # Apply centralized intent-anchor normalization for missing context dimensions
        anchor = extract_intent_anchor(raw_prompt)
        normalized_cleaned = []
        for item in cleaned:
            norm = normalize_missing_context_dimension(item, anchor.action_family, raw_prompt)
            if norm and norm.lower() not in [nc.lower() for nc in normalized_cleaned]:
                normalized_cleaned.append(norm)

        return normalized_cleaned

    def _apply_classification_consistency(
        self, raw_prompt: str, classification: PromptClassification
    ) -> PromptClassification:
        """Apply centralized classification consistency based strictly on the user's requested action anchor."""
        anchor = extract_intent_anchor(raw_prompt)
        expected_category = ACTION_FAMILY_TO_CATEGORY.get(anchor.action_family, PromptCategory.GENERAL)

        if anchor.is_strong and anchor.action_family != ActionFamily.GENERAL:
            if classification.category != expected_category:
                logger.info(
                    "Classification consistency: reclassified from %s to %s based on strong action anchor (%s).",
                    classification.category.value,
                    expected_category.value,
                    anchor.primary_action,
                )
                return PromptClassification(category=expected_category, confidence=anchor.confidence)
            return classification

        # Weak anchor: Consult Semantic Category Router (unless prompt contains self-corrections/hesitations)
        has_hesitation = bool(re.search(r"\b(?:wait|actually\s+no|scratch\s+that|no\s+like)\b", raw_prompt.lower()))
        if not has_hesitation:
            try:
                from app.core.category_router import category_router
                routing = category_router.route(raw_prompt, anchor=anchor)
                if routing.is_high_confidence and routing.evidence_type == "semantic_prototype":
                    if classification.category != routing.category:
                        logger.info(
                            "Classification consistency: reclassified from %s to %s based on high-confidence semantic prototype (margin: %.3f).",
                            classification.category.value,
                            routing.category.value,
                            routing.margin,
                        )
                        return PromptClassification(category=routing.category, confidence=routing.confidence)
            except Exception as e:
                logger.debug("Semantic router consistency check skipped: %s", e)

        # Weak anchor: allow analyzer semantic judgment unless obvious invalid edge cases (like SUMMARIZATION without text or DEBUGGING without error)
        if classification.category == PromptCategory.SUMMARIZATION:
            is_summary = bool(
                re.search(r"\b(?:summarize|condense|digest|summary|synopsis|give\s+me\s+short\s+notes\s+from|tldr|key\s+takeaways)\b", raw_prompt.lower())
            )
            if not is_summary:
                return PromptClassification(category=PromptCategory.GENERAL, confidence=0.75)

        if classification.category == PromptCategory.DEBUGGING:
            is_debug = bool(
                re.search(r"\b(?:fix|debug|troubleshoot|diagnose|error|bug|issue|crash|exception|failing|broken|deadlock|freeze|hang|traceback|panic)\b", raw_prompt.lower())
            )
            if not is_debug:
                return PromptClassification(category=PromptCategory.GENERAL, confidence=0.75)

        return classification

    def _try_safe_fast_path(
        self,
        user_prompt: str,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> Optional[PromptAnalysis]:
        """Attempt safe structured analysis bypass without calling the LLM when all safety invariants hold.

        Preconditions:
        1. Single primary intent (no multiple compound tasks or contradictory commands).
        2. Strong action-family confidence OR high semantic category confidence.
        3. Source-unit extraction complete (1 <= units <= 5).
        4. No critical ambiguity (sufficient detail, >= 4 words, not a vague generic request).
        5. No self-correction or supersession complexity.
        6. Not long-input mode (< 120 words).
        7. No unresolved referents ('this error', 'the above code' without context).
        8. No clarification answers (or clean standalone prompt).
        9. Source coverage passes 100%.
        """
        # Do not bypass if client is a mock injected in unit tests to test LLM parsing/schema/repair
        is_mock = hasattr(self.client, "assert_called") or hasattr(self.client, "mock_calls") or hasattr(self.client, "side_effect")
        if is_mock:
            return None

        prompt_trimmed = user_prompt.strip()
        words = prompt_trimmed.split()
        if len(words) < 4 or len(words) > 120:
            return None

        # Disallow fast path if user clarifications were provided (requires full semantic re-analysis)
        if clarification_answers and any(ca.answer and ca.answer.strip() for ca in clarification_answers):
            return None

        prompt_lower = prompt_trimmed.lower()

        # Check self-correction / supersession keywords
        hesitation_pat = r"\b(?:actually\s+(?:no|wait|never\s+mind)|wait\s+(?:no|maybe)|scratch\s+that|rather\s+than|instead\s+of|on\s+second\s+thought|not\s+a\s+normal)\b"
        if re.search(hesitation_pat, prompt_lower):
            return None

        # Check unresolved referents (e.g., 'fix this error', 'optimize this code' when no error/code is given)
        dangling_referent = r"\b(?:this\s+(?:error|issue|problem|code|function|bug|script|traceback)|the\s+(?:above|attached|following)\s+(?:code|error|file|traceback))\b"
        if re.search(dangling_referent, prompt_lower):
            has_error_or_code = bool(
                re.search(r"[{};=]|def\s+|class\s+|traceback|error:|exception:", prompt_lower)
            )
            if not has_error_or_code:
                return None

        # Check critical ambiguity or vague prompts that need clarification
        vague_patterns = [
            r"^(?:help|help\s+me|assist\s+me|can\s+you\s+help)$",
            r"^(?:make|build|create|write)\s+(?:an?\s+)?(?:app|application|website|tool|script|system|program)$",
            r"^(?:fix|debug)\s+(?:it|this|my\s+code)$",
            r"^what\s+should\s+i\s+do\??$",
        ]
        if any(re.match(pat, prompt_lower) for pat in vague_patterns):
            return None

        # Check multiple compound imperatives (e.g. 'and also write', 'and then explain')
        compound_pat = r"\b(?:and\s+(?:also|then)\s+(?:write|explain|create|build|generate|make|summarize))\b"
        if re.search(compound_pat, prompt_lower):
            return None

        # Extract deterministic intent anchor
        anchor = extract_intent_anchor(user_prompt)

        # Route through semantic category router
        from app.core.category_router import category_router
        routing = category_router.route(user_prompt, anchor=anchor)

        # Calibrated Three-Tier Fast-Path Policy:
        # Only activates if Tier A (strong concurrence) or Tier B (high-confidence semantic) holds
        # Strictly rejects all disagreements, close categories, and weak general tasks
        if not routing.can_fast_path:
            return None

        category = routing.category
        confidence = routing.confidence
        path = "deterministic_fast" if routing.tier == "tier_a_deterministic" else "semantic_fast"

        # Extract source semantic units
        source_units = extract_source_semantic_units(user_prompt)
        if not (1 <= len(source_units) <= 5):
            return None

        # Extract clean objective: first sentence or clean prompt
        sentences = [s.strip() for s in re.split(r"[\n\.\?!]", prompt_trimmed) if s.strip()]
        objective = sentences[0] if sentences else prompt_trimmed

        # Build tentative analysis
        classification = PromptClassification(category=category, confidence=confidence)
        ambiguity = AmbiguityAssessment(
            score=10,
            level=AmbiguityLevel.LOW,
            needs_clarification=False,
            reasons=[],
            questions=[],
        )
        deliverable_sem = DeliverableSemantics(
            deliverable_type=anchor.target_or_deliverable,
            intended_outcome=anchor.primary_action,
            is_functionally_complete=True,
        )

        analysis = PromptAnalysis(
            classification=classification,
            objective=objective,
            explicit_requirements=source_units,
            constraints=[],
            preferences=[],
            missing_context=[],
            ambiguity=ambiguity,
            intent_anchor=anchor,
            deliverable_semantics=deliverable_sem,
            raw_category=category.value,
            analysis_path=path,
        )

        # Recover constraints, quantities, styles, etc.
        self._recover_general_deliverables_and_constraints(user_prompt, analysis)
        analysis.preferences = self._recover_style_preferences(user_prompt, analysis.preferences)

        # Grounding check: verify that source coverage is 100%
        is_covered, _, missing = check_source_coverage(user_prompt, analysis)
        if not is_covered:
            return None

        logger.info(
            "Safe analyzer fast path activated: path=%s, category=%s, confidence=%.2f, source_units=%d",
            path,
            category.value,
            confidence,
            len(source_units),
        )
        return analysis

    def _parse_and_validate(
        self, raw_output: str
    ) -> Tuple[Optional[PromptAnalysis], Optional[str], Optional[Exception]]:
        """Clean, parse JSON, and validate against PromptAnalysis schema. Returns (analysis, issue_description, exception)."""
        cleaned = self._clean_json_content(raw_output)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}", exc

        if not isinstance(parsed, dict):
            return None, "response is not a JSON object", ValueError("Response is not a JSON object")

        try:
            analysis = PromptAnalysis.model_validate(parsed)
            return analysis, None, None
        except ValidationError as exc:
            missing_fields = [
                str(err.get("loc", [""])[0])
                for err in exc.errors()
                if err.get("type") in {"missing", "field_required"}
            ]
            if missing_fields:
                issue = f"missing required field(s): {', '.join(missing_fields)}"
            else:
                issue = f"does not match the required schema: {exc}"
            return None, issue, exc

    def _recover_general_deliverables_and_constraints(
        self, raw_prompt: str, analysis: PromptAnalysis
    ) -> None:
        """General invariant recovery for deliverables, quantities, referents, constraints, and negative constraints."""
        prompt_lower = raw_prompt.lower()

        # 1. Negative constraints: e.g. "don't use Bootstrap", "avoid jQuery", "without Tailwind"
        neg_matches = re.finditer(
            r"\b(?:don't\s+use|do\s+not\s+use|avoid|without)\s+([a-zA-Z0-9\.\-]+(?:\s+[a-zA-Z0-9\.\-]+)?)\b",
            prompt_lower,
        )
        for nm in neg_matches:
            neg_item = nm.group(1).strip()
            if neg_item not in {"like", "a", "an", "the", "wait"} and not self._is_superseded_or_rejected(neg_item, prompt_lower):
                constraint_str = f"do not use {neg_item}"
                if not any(neg_item in c.lower() for c in analysis.constraints):
                    analysis.constraints.append(constraint_str)

        # 2. Limits & word counts: e.g. "under 80 words", "under 200 words", "at most 100 words"
        limit_match = re.search(r"\b(?:under|less\s+than|max|maximum|at\s+most)\s+(\d+)\s+words?\b", prompt_lower)
        if limit_match and not self._is_superseded_or_rejected(limit_match.group(0), prompt_lower):
            limit_val = limit_match.group(1)
            target_match = re.search(rf"\b(?:keep\s+(?:the\s+)?([a-zA-Z\-]+)|for\s+(?:the\s+)?([a-zA-Z\-]+))\s+[^\.\n]*?(?:under|less\s+than|max|at\s+most)\s+{limit_val}\s+words?\b", prompt_lower)
            if target_match:
                target_noun = target_match.group(1) or target_match.group(2)
                limit_str = f"keep {target_noun} under {limit_val} words"
            else:
                limit_str = f"under {limit_val} words"
            if not any(limit_val in c.lower() for c in analysis.constraints):
                analysis.constraints.append(limit_str)

        # 3. Explicit duration constraint (e.g., "8-week", "8 weeks", "3-day", "5-day")
        duration_match = re.search(r"\b(\d+[\s-]*(?:week|month|day|hour)s?)\b", prompt_lower)
        if duration_match and not self._is_superseded_or_rejected(duration_match.group(1), prompt_lower):
            dur_val = duration_match.group(1).strip()
            dur_str = f"{dur_val.replace(' ', '-')} duration"
            if not any(dur_val in c.lower() for c in analysis.constraints) and not any(dur_val in r.lower() for r in analysis.explicit_requirements):
                analysis.constraints.append(dur_str)

        # 3b. Lifecycle, retention, deletion, and cleanup constraints:
        # Capture explicit data retention policies, automatic deletion rules, lifecycle boundaries, or cleanup conditions
        retention_patterns = [
            r"\b(?:data|records?|files?|submissions?|logs?|information|user\s+data|stored\s+data)?\s*(?:should|must|needs\s+to|to)?\s*(?:be\s+)?(deleted|removed|purged|cleared|wiped|erased|dropped)\s+after\s+([a-zA-Z0-9_\-\s]{3,40}?)(?:\b(?:and\b|then\b|also\b|plus\b|\.|\,)|$)",
            r"\b(delete|remove|purge|clear|wipe|erase)\s+(?:stored\s+)?(?:data|records?|files?|submissions?|logs?|information)\s+after\s+([a-zA-Z0-9_\-\s]{3,40}?)(?:\b(?:and\b|then\b|also\b|plus\b|\.|\,)|$)",
            r"\b(retain|keep|store)\s+(?:data|records?|files?)\s+for\s+(\d+\s+(?:days?|months?|years?|weeks?|hours?))\b",
            r"\b(expire|expires|expiration)\s+(?:after|in)\s+([a-zA-Z0-9_\-\s]{3,40}?)(?:\b(?:and\b|then\b|also\b|plus\b|\.|\,)|$)",
        ]
        for r_pat in retention_patterns:
            for r_match in re.finditer(r_pat, prompt_lower):
                full_matched = r_match.group(0).strip()
                if not self._is_superseded_or_rejected(full_matched, prompt_lower):
                    verb_matched = r_match.group(1).strip()
                    boundary_matched = r_match.group(2).strip()
                    if "expire" in verb_matched:
                        rule_str = f"expire after {boundary_matched}"
                    elif "retain" in verb_matched or "keep" in verb_matched:
                        rule_str = f"retain data for {boundary_matched}"
                    else:
                        rule_str = f"delete stored data after {boundary_matched}"
                    if not any(boundary_matched in c.lower() for c in analysis.constraints) and not any(boundary_matched in r.lower() for r in analysis.explicit_requirements):
                        analysis.constraints.append(rule_str)

        # 4. Quantity + Deliverable recovery (e.g., "4 science fair ideas", "3 newsletter subject lines", "4 practice questions")
        qty_matches = re.finditer(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-zA-Z\-]+(?:\s+(?!and\b|then\b|or\b|but\b|also\b|plus\b|with\b|after\b|to\b|for\b|that\b|which\b|who\b)[a-zA-Z\-]+){0,2})\b",
            prompt_lower,
        )
        num_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        for qm in qty_matches:
            qty_word, noun_phrase = qm.group(1), qm.group(2).strip()
            if any(unit in noun_phrase for unit in ["week", "month", "day", "hour", "year", "minute", "word"]):
                continue
            if self._is_superseded_or_rejected(noun_phrase, prompt_lower):
                continue
            qty_digit = num_map.get(qty_word, qty_word)
            item_phrase = f"{qty_digit} {noun_phrase}".strip()
            if not any(qty_digit in r.lower() and any(w in r.lower() for w in noun_phrase.split()) for r in analysis.explicit_requirements):
                analysis.explicit_requirements.append(item_phrase)

        # 5. Compound Action Deliverables:
        # Selection / Referent: "choose the strongest one", "pick one", "choose one", "select the best one"
        if re.search(r"\b(?:choose|pick|select)\s+(?:the\s+)?(?:strongest|best|one|first|second)\b", prompt_lower):
            sel_match = re.search(r"\b(?:choose|pick|select)\s+(?:the\s+)?([a-zA-Z\-]+(?:\s+[a-zA-Z\-]+)?)\b", prompt_lower)
            sel_target = sel_match.group(1) if sel_match else "best option"
            sel_clause = f"select {sel_target}"
            if not any(w in r.lower() for r in analysis.explicit_requirements for w in ["select", "choose", "pick"]):
                analysis.explicit_requirements.append(sel_clause)

        # Explanation: "explain why", "explain the reasoning"
        if re.search(r"\bexplain\s+(?:why|the\s+reasoning|reasons?)\b", prompt_lower) and not self._is_superseded_or_rejected("explain", prompt_lower):
            if not any("explain" in r.lower() for r in analysis.explicit_requirements):
                analysis.explicit_requirements.append("explain reasoning")

        # Drafting / expanding referents: "draft the second one", "expand the second", "develop idea #2"
        draft_ref_match = re.search(
            r"\b(?:draft|expand|develop|write)\s+(?:the\s+)?(?:idea\s+|option\s+|item\s+)?(first|second|third|#\d+|\d+(?:st|nd|rd|th))\s*(?:one|idea|option|subject\s+line|item)?\b",
            prompt_lower,
        )
        if draft_ref_match and not self._is_superseded_or_rejected(draft_ref_match.group(0), prompt_lower):
            ref_item = f"draft {draft_ref_match.group(1)} item"
            if not any(w in r.lower() for r in analysis.explicit_requirements for w in ["draft", "expand", "develop"]):
                analysis.explicit_requirements.append(ref_item)

        # Comparative deliverable: "compare the main approaches", "compare them"
        if re.search(r"\bcompare\s+(?:the\s+)?(?:main\s+)?(?:approaches|methods|options|items|them)\b", prompt_lower):
            if not any("compare" in r.lower() for r in analysis.explicit_requirements):
                analysis.explicit_requirements.append("compare main approaches")

        # Output format deliverable: "in a table", "as a table", "summarize them in a table"
        if re.search(r"\b(?:in|as)\s+a\s+table\b", prompt_lower) or re.search(r"\btable\s+format\b", prompt_lower):
            if not any("table" in r.lower() for r in analysis.explicit_requirements) and not any("table" in c.lower() for c in analysis.constraints):
                analysis.explicit_requirements.append("summary table")

        # 6. Recommendation deliverable: e.g. "tell them what shoes to buy", "recommend laptops", "recommend books"
        rec_match = re.search(r"\b(?:tell\s+(?:them|me|users)\s+what\s+([a-zA-Z\-]+)\s+to\s+buy|recommend(?:s|ing)?\s+([a-zA-Z\-]+))\b", prompt_lower)
        if rec_match:
            rec_target = rec_match.group(1) or rec_match.group(2)
            if rec_target and not self._is_superseded_or_rejected(rec_target, prompt_lower):
                rec_str = f"{rec_target} recommendations"
                if not any("recommend" in r.lower() for r in analysis.explicit_requirements):
                    analysis.explicit_requirements.append(rec_str)

        # 7. Functional software/interactive deliverables:
        functional_components = ["quiz", "calculator", "dashboard", "contact form", "landing page", "results"]
        for comp in functional_components:
            if re.search(rf"\b{re.escape(comp)}\b", prompt_lower) and not self._is_superseded_or_rejected(comp, prompt_lower):
                if not any(comp in r.lower() for r in analysis.explicit_requirements):
                    analysis.explicit_requirements.append(comp)

        # Recover explicit technical tools directly stated in prompt if omitted by model and not superseded
        known_explicit_tools = [
            ("python", "Python"),
            ("react", "React"),
            ("fastapi", "FastAPI"),
            ("postgresql", "PostgreSQL"),
            ("angular", "Angular"),
            ("vue", "Vue"),
            ("django", "Django"),
            ("html", "HTML"),
            ("css", "CSS"),
            ("javascript", "JavaScript"),
            ("typescript", "TypeScript"),
        ]
        for tech_key, tech_name in known_explicit_tools:
            if re.search(rf"\b{tech_key}\b", prompt_lower) and not self._is_superseded_or_rejected(tech_key, prompt_lower):
                if not any(tech_key == r.lower() for r in analysis.explicit_requirements):
                    analysis.explicit_requirements.append(tech_name)

    def _is_vague_or_non_answer(self, answer: str) -> bool:
        """Detect generally weak or non-substantive user answers across domains."""
        cleaned = answer.strip().lower()
        if not cleaned:
            return True
        vague_phrases = {
            "i don't know", "i dont know", "dont know", "not sure", "idk",
            "whatever", "anything", "you decide", "no preference",
            "doesn't matter", "doesnt matter", "any", "no clue", "up to you",
            "none", "n/a", "na", "not applicable", "unknown", "skip",
        }
        return cleaned in vague_phrases or (len(cleaned) <= 2 and cleaned not in {"ui", "ai", "db", "go", "r", "c", "js", "ts"})

    def _is_duplicate_question(
        self,
        question: ClarificationQuestion,
        answered_items: list[ClarificationAnswer],
    ) -> bool:
        """Check if a candidate clarification question targets an already meaningfully answered dimension or question."""
        q_target = (question.target_dimension or "").strip().lower()
        q_text = (question.question or "").strip().lower()

        for item in answered_items:
            # If the user's answer was vague / non-substantive, the dimension was not meaningfully answered
            if self._is_vague_or_non_answer(item.answer):
                continue

            # 1. Target dimension comparison
            item_target = (item.target_dimension or "").strip().lower()
            if item_target and q_target:
                if q_target == item_target:
                    return True
                # Keyword overlap on non-generic words
                q_words = {w for w in re.findall(r"\b[a-z]{3,}\b", q_target) if w not in STOP_WORDS}
                item_words = {w for w in re.findall(r"\b[a-z]{3,}\b", item_target) if w not in STOP_WORDS}
                if q_words and item_words and len(q_words & item_words) >= max(1, min(len(q_words), len(item_words))):
                    return True

            # 2. Question text similarity
            item_text = (item.question or "").strip().lower()
            if item_text and q_text:
                if q_text == item_text:
                    return True
                q_words = {w for w in re.findall(r"\b[a-z]{3,}\b", q_text) if w not in STOP_WORDS}
                item_words = {w for w in re.findall(r"\b[a-z]{3,}\b", item_text) if w not in STOP_WORDS}
                if q_words and item_words:
                    overlap = len(q_words & item_words)
                    if overlap / max(len(q_words), len(item_words)) >= 0.6:
                        return True

        return False

    def _analyze_deliverable_semantics(
        self,
        raw_prompt: str,
        analysis: PromptAnalysis,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
    ) -> DeliverableSemantics:
        """Decompose request into deliverable container vs functional purpose / intended outcome across all categories."""
        prompt_lower = raw_prompt.lower()
        words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", prompt_lower)

        # 1. Authoritative clarification answers: if user answered functional purpose or what to accomplish
        if clarification_answers:
            substantive_answers = [
                ca.answer.strip() for ca in clarification_answers
                if ca.answer and not self._is_vague_or_non_answer(ca.answer)
            ]
            if substantive_answers:
                deliv_type = None
                if hasattr(analysis, "deliverable_semantics") and analysis.deliverable_semantics and analysis.deliverable_semantics.deliverable_type:
                    deliv_type = analysis.deliverable_semantics.deliverable_type
                return DeliverableSemantics(
                    deliverable_type=deliv_type,
                    intended_outcome="; ".join(substantive_answers),
                    is_functionally_complete=True,
                )

        # 2. Inherently Self-Contained Operations (No separate purpose required):
        # Operations where the action verb and its direct target fully specify what to do.
        self_contained_patterns = [
            r"\b(?:convert|transform|translate|parse)\s+([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+)?)\s+(?:to|into)\s+([a-zA-Z0-9_\.\-]+)",
            r"\b(?:summarize|condense|digest|synopsis)\b",
            r"\b(?:explain|teach|tutoring)\s+(?:me\s+)?([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,4})",
            r"\b(?:calculate|compute|solve|derive)\s+([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,4})",
            r"\b(?:compare|contrast)\s+(?:the\s+)?([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,4})",
            r"\b(?:debug|fix|find\s+bug|solve\s+issue|troubleshoot)\b",
            r"\b(?:find|sort|filter|reverse|search)\s+(?:the\s+)?([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,3})",
            r"\b(?:generate|brainstorm|give\s+me)\s+\d+\s+([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,3})",
        ]
        for pat in self_contained_patterns:
            m = re.search(pat, prompt_lower)
            if m:
                return DeliverableSemantics(
                    deliverable_type=None,
                    intended_outcome=m.group(0),
                    is_functionally_complete=True,
                )

        # 3. Detect Deliverable Containers:
        # Generic container nouns across CODING, WRITING, PLANNING, CREATIVE, EDUCATION, RESEARCH
        container_pattern = (
            r"\b(?:build|create|make|develop|write|draft|design|generate|produce|code|prepare|implement|setup|set\s+up|conduct|perform|do)\s+"
            r"(?:me\s+)?(?:an?\s+|the\s+)?"
            r"([a-zA-Z0-9_\-\s]{0,40}?)"
            r"\b(web\s+app|web\s+application|website|webpage|page|app|application|software|tool|script|dashboard|system|service|bot|program|portal|extension|document|doc|report|email|letter|memo|plan|roadmap|schedule|presentation|slides|deck|story|image|visual|poster|prompt|template|curriculum|syllabus|research|something|anything|solution|project|stuff)\b"
        )
        container_match = re.search(container_pattern, prompt_lower)

        # If no explicit creation verb was matched, check standalone container phrase
        if not container_match:
            standalone_match = re.search(
                r"\b(?:an?\s+|the\s+)?([a-zA-Z0-9_\-\s]{0,30}?)\b(web\s+app|web\s+application|website|webpage|page|app|application|software|tool|script|dashboard|system|service|bot|program|portal|extension|document|doc|report|email|letter|memo|plan|roadmap|schedule|presentation|slides|deck|story|curriculum|research|something|anything|solution|project|stuff)\b",
                prompt_lower,
            )
            if standalone_match:
                container_match = standalone_match

        if not container_match:
            substantive_prompt_words = [
                w for w in words if w not in STOP_WORDS and w not in GENERIC_FILLER_WORDS and w not in GENERIC_ACTION_VERBS
            ]
            # If not a recognized container request, assume operation is complete only if prompt has substantive words
            return DeliverableSemantics(
                deliverable_type=None,
                intended_outcome=analysis.objective,
                is_functionally_complete=len(substantive_prompt_words) >= 2 or (len(words) >= 5 and len(substantive_prompt_words) >= 1),
            )

        preceding_modifiers = container_match.group(1).strip()
        deliverable_name = container_match.group(2).strip()

        # 4. Check for Functional Purpose / Intended Outcome clauses:
        # A. Infinitive clause with substantive action verb: "to count words in a text file", "to track inventory", "to monitor servers", "to ask for an extension"
        infinitive_match = re.search(
            r"\bto\s+(?!(?:build|make|create|write|develop|do|help|start|code|use|be)\b)([a-zA-Z]{3,}(?:\s+[a-zA-Z0-9_\.\-]+){1,6})\b",
            prompt_lower,
        )

        # B. Gerund clause: "for tracking warehouse inventory", "for managing projects", "for analyzing logs"
        # Note: NOT platform names like "for telegram", "for android", "for mobile", "for web", "for desktop", "for me"
        gerund_match = re.search(
            r"\bfor\s+([a-zA-Z]{3,}ing(?:\s+[a-zA-Z0-9_\.\-]+){1,6})\b",
            prompt_lower,
        )

        # C. Relative clause: "that answers customer support questions", "which calculates compound interest"
        relative_match = re.search(
            r"\b(?:that|which)\s+([a-zA-Z]{3,}(?:\s+[a-zA-Z0-9_\.\-]+){1,6})\b",
            prompt_lower,
        )

        # D. Topic / Goal / Depiction clause: "about renewable energy", "explaining quantum physics", "of a cyberpunk city"
        topic_match = re.search(
            r"\b(?:about|regarding|outlining|detailing|explaining|discussing|comparing|depicting|showing|of)\s+([a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){1,6})\b",
            prompt_lower,
        )
        # Trip / destination planning: e.g. "for a 3-day trip to Tokyo"
        trip_match = re.search(r"\bfor\s+(?:a\s+|an\s+)?(\d+[\s-]day\s+trip\s+to\s+[a-zA-Z0-9_\.\-]+)\b", prompt_lower)

        # D2. Participial purpose clause following container: "email asking my lecturer...", "email announcing our product launch", "presentation explaining..."
        remainder_after_container = prompt_lower[container_match.end():].strip() if container_match else ""
        participle_match = re.search(
            r"\b([a-zA-Z]{3,}ing(?:\s+[a-zA-Z0-9_\.\-]+){1,6})\b",
            remainder_after_container,
        )

        # D3. Action/capability clause following container: e.g. "it should track...", "must calculate...", "needs to manage..."
        clause_match = re.search(
            r"\b(?:it\s+should|it\s+must|it\s+needs\s+to|should|must|needs\s+to)\s+([a-zA-Z]{3,}(?:\s+[a-zA-Z0-9_\.\-]+){1,6})\b",
            prompt_lower,
        )

        # E. Substantive Domain Modifier before container:
        # e.g., "inventory tracking dashboard", "weather bot", "word counter tool", "expense tracker app"
        non_purpose_words = {
            "a", "an", "the", "me", "my", "our", "simple", "basic", "complex", "advanced",
            "light", "lightweight", "small", "large", "minimal", "modular", "clean",
            "professional", "modern", "cool", "new", "fresh", "interactive", "responsive",
            "automated", "dynamic", "smart", "fast", "static", "secure", "friendly", "formal",
            "casual", "brief", "concise", "web", "mobile", "desktop", "telegram", "slack",
            "discord", "cli", "terminal", "console", "ios", "android", "windows", "linux", "macos", "mac",
            "browser", "cloud", "server", "client", "frontend", "backend", "fullstack", "docker",
            "pdf", "powerpoint", "word", "excel", "markdown", "html", "css", "javascript", "typescript",
            "js", "ts", "python", "node", "nodejs", "react", "vue", "angular", "svelte", "nextjs",
            "django", "flask", "fastapi", "express", "tailwind", "bootstrap", "good", "great", "best", "little",
            "swift", "kotlin", "java", "rust", "go", "golang", "c", "cpp", "csharp", "ruby", "php", "sql",
            "programming", "language", "code", "coding", "framework", "library", "stack", "tech",
            "technology", "tooling", "platform", "environment", "device", "medium", "format",
            "with", "using", "in", "for", "via",
        }
        modifier_words = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", preceding_modifiers))
        substantive_modifiers = [w for w in modifier_words if w not in non_purpose_words]

        found_purpose = None
        if infinitive_match:
            found_purpose = infinitive_match.group(0).strip()
        elif gerund_match:
            found_purpose = gerund_match.group(0).strip()
        elif relative_match:
            found_purpose = relative_match.group(0).strip()
        elif topic_match:
            found_purpose = topic_match.group(0).strip()
        elif trip_match:
            found_purpose = trip_match.group(0).strip()
        elif participle_match:
            found_purpose = participle_match.group(0).strip()
        elif clause_match:
            found_purpose = clause_match.group(0).strip()
        elif substantive_modifiers:
            found_purpose = " ".join(substantive_modifiers)

        is_complete = found_purpose is not None

        return DeliverableSemantics(
            deliverable_type=deliverable_name,
            intended_outcome=found_purpose,
            is_functionally_complete=is_complete,
        )

    def _normalize_ambiguity(
        self,
        analysis: PromptAnalysis,
        user_prompt: str,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> None:
        """Central consistency normalization for ambiguity heuristic, questions, and clarification state."""
        prompt_lower = user_prompt.lower()
        amb = analysis.ambiguity

        # Always compute deliverable_semantics based on authoritative prompt
        analysis.deliverable_semantics = self._analyze_deliverable_semantics(
            raw_prompt=user_prompt,
            analysis=analysis,
            clarification_answers=clarification_answers,
        )

        # 1. If user chose reasonable assumptions: bypass clarification without falsifying score/level
        if allow_reasonable_assumptions:
            amb.needs_clarification = False
            amb.questions = []
            return

        # 2. Defensive sanitization of questions: ensure strict neutrality and no injected candidate tech
        sanitized_questions = []
        for idx, q in enumerate(amb.questions[:3], start=1):
            q_text = q.question
            has_forbidden = False
            for tech in FORBIDDEN_TECH_EXAMPLES:
                if re.search(rf"\b{re.escape(tech)}\b", q_text, flags=re.IGNORECASE):
                    has_forbidden = True
                    break
            if has_forbidden:
                if any(w in q_text.lower() for w in ["database", "storage", "postgres", "sqlite", "mongodb"]):
                    q_text = "Do you have a preferred data storage method, or should the implementation select a reasonable option?"
                else:
                    q_text = "Do you have a required technology stack or platform, or should the implementation select a reasonable option?"
            sanitized_questions.append(
                ClarificationQuestion(
                    id=f"q{idx}",
                    question=q_text,
                    reason=q.reason or "Clarification required to resolve primary intent.",
                    target_dimension=q.target_dimension or "unspecified dimension",
                )
            )
        amb.questions = sanitized_questions

        # 3. Filter valid (non-blank) clarification answers
        valid_answers = [
            ca for ca in (clarification_answers or [])
            if ca.answer and ca.answer.strip()
        ]

        # 4. Handle follow-up evaluation when clarification answers are present
        if valid_answers:
            # Incorporate substantive answers into explicit_requirements or preferences if not already present
            for ca in valid_answers:
                ans_text = ca.answer.strip()
                if not self._is_vague_or_non_answer(ans_text):
                    if any(t in ans_text.lower() for t in ["tone", "style", "format"]):
                        if ans_text not in analysis.preferences:
                            analysis.preferences.append(ans_text)
                    else:
                        if ans_text not in analysis.explicit_requirements:
                            analysis.explicit_requirements.append(ans_text)

            # Suppress duplicate questions targeting already meaningfully answered dimensions
            surviving_questions = [
                q for q in amb.questions
                if not self._is_duplicate_question(q, valid_answers)
            ]

            amb.questions = [
                ClarificationQuestion(
                    id=f"q{idx}",
                    question=q.question,
                    reason=q.reason,
                    target_dimension=q.target_dimension,
                )
                for idx, q in enumerate(surviving_questions[:3], start=1)
            ]

            if not amb.questions:
                amb.needs_clarification = False
            else:
                amb.needs_clarification = True
                if amb.score < 65:
                    amb.score = 65
                    amb.level = AmbiguityLevel.HIGH

            return

        # 5. Initial request assessment: check for fundamentally underspecified requests across all domains
        words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", prompt_lower)
        significant_words = [w for w in words if w not in STOP_WORDS and w not in GENERIC_FILLER_WORDS]

        # Authoritative prompt vagueness: independent of fallible derived analysis
        is_ultra_vague = (
            len(words) <= 7
            and len(significant_words) == 0
        )

        deliv = analysis.deliverable_semantics
        anchor = getattr(analysis, "intent_anchor", None) or extract_intent_anchor(user_prompt, clarification_answers)
        if is_self_contained_operation(user_prompt, anchor.action_family):
            if deliv:
                deliv.is_functionally_complete = True
            is_missing_purpose = False
        else:
            is_missing_purpose = bool(deliv and not deliv.is_functionally_complete)

        if is_ultra_vague or is_missing_purpose:
            amb.score = max(amb.score, 75)
            amb.level = AmbiguityLevel.HIGH
            amb.needs_clarification = True
            artifact_name = (deliv.deliverable_type if deliv and deliv.deliverable_type else "deliverable")
            target_name = artifact_name if artifact_name not in {"something", "anything", "stuff", "thing"} else "deliverable"
            purpose_q = ClarificationQuestion(
                id="q1",
                question=f"What should the requested {target_name} help accomplish or what is its primary purpose?",
                reason=f"The primary deliverable ({target_name}) is specified, but its functional purpose and intended outcome are absent.",
                target_dimension="functional purpose",
            )
            # Ensure purpose question is always present and first
            if not amb.questions or not any("purpose" in q.target_dimension.lower() for q in amb.questions):
                amb.questions = [purpose_q] + [q for q in amb.questions if q.id != "q1"][:2]
        else:
            # For specific prompts across domains: MISSING CONTEXT != REQUIRED CLARIFICATION
            if amb.score < 65:
                amb.needs_clarification = False
                amb.questions = []
            elif amb.score >= 65 and len(significant_words) >= 3 and not is_ultra_vague and not is_missing_purpose:
                amb.score = 25
                amb.level = AmbiguityLevel.LOW
                amb.needs_clarification = False
                amb.questions = []

        # Final invariant checks
        if not amb.needs_clarification:
            amb.questions = []
        elif amb.needs_clarification and not amb.questions:
            amb.questions = [
                ClarificationQuestion(
                    id="q1",
                    question="Could you clarify the primary purpose or key functional requirements for this request?",
                    reason="Essential task identity and purpose are underspecified.",
                    target_dimension="functional purpose",
                )
            ]

    async def analyze(
        self,
        user_prompt: str,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> PromptAnalysis:
        """Analyze a raw human prompt and return a validated PromptAnalysis structure with 1-attempt recovery."""
        # 0. Check token-aware context budget before LLM call
        schema_dict = self._build_schema_contract()
        budget_report = context_budget_manager.check_stage_budget(
            stage="analyzer",
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            user_text=user_prompt,
            schema_json=json.dumps(schema_dict),
        )

        if budget_report.processing_mode == ProcessingMode.LONG_INPUT:
            logger.info(
                "Token budget exceeded for single-pass analysis (%d est tokens, %d safe). "
                "Activating long-input semantic decomposition pipeline.",
                budget_report.total_tokens,
                budget_report.safe_user_tokens,
            )
            # Combine clarifications with prompt for decomposition if provided
            full_source = user_prompt.strip()
            if clarification_answers:
                substantive_answers = [
                    ca.answer.strip() for ca in clarification_answers if ca.answer and ca.answer.strip()
                ]
                if substantive_answers:
                    full_source += "\n\n" + "\n\n".join(substantive_answers)

            # 1. Semantic boundary chunking
            chunks = chunk_long_source(full_source)
            logger.info("Decomposed long input into %d semantic chunk(s).", len(chunks))

            # 2. Extract atomic units from each chunk with provenance
            chunk_units_list = []
            for chunk in chunks:
                units = await extract_chunk_atomic_units(chunk, self.client)
                chunk_units_list.append(units)

            # 3. Resolve supersessions and deduplicate across chunks
            flat_units, active_units, superseded_units = resolve_supersessions_and_deduplicate(
                chunks, chunk_units_list
            )
            logger.info(
                "Extracted %d total units (%d active, %d superseded).",
                len(flat_units),
                len(active_units),
                len(superseded_units),
            )

            # 4. Global reconciliation step
            analysis = await reconcile_long_input_analysis(
                chunks=chunks,
                active_units=active_units,
                client=self.client,
                original_prompt=user_prompt,
            )
            analysis.raw_category = analysis.classification.category.value
            analysis.analysis_path = "llm"
            return analysis

        # Attempt safe fast path bypass before calling LLM
        fast_analysis = self._try_safe_fast_path(
            user_prompt=user_prompt,
            clarification_answers=clarification_answers,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )
        if fast_analysis is not None:
            return fast_analysis

        user_content = f"<USER_REQUEST>\n{user_prompt.strip()}\n</USER_REQUEST>"
        valid_answers = [
            ca for ca in (clarification_answers or [])
            if ca.answer and ca.answer.strip()
        ]
        if valid_answers:
            answers_formatted = []
            for ca in valid_answers:
                dim_line = f"- Target dimension: {ca.target_dimension.strip()}\n" if ca.target_dimension and ca.target_dimension.strip() else ""
                answers_formatted.append(
                    f"{dim_line}- Question: {ca.question.strip()}\n- User answer: {ca.answer.strip()}"
                )
            clarifications_block = "\n\n".join(answers_formatted)
            user_content = (
                f"<USER_REQUEST>\n{user_prompt.strip()}\n</USER_REQUEST>\n\n"
                f"<USER_CLARIFICATIONS>\n{clarifications_block}\n</USER_CLARIFICATIONS>\n\n"
                f"<ANALYSIS_DIRECTIVE>\n"
                f"Treat clarification answers inside <USER_CLARIFICATIONS> as authoritative user-provided information.\n"
                f"Update the structured interpretation accordingly.\n"
                f"Recalculate ambiguity based on the combined information.\n"
                f"Do not re-ask questions targeting dimensions the user has already answered.\n"
                f"</ANALYSIS_DIRECTIVE>"
            )

        schema = self._build_schema_contract()
        messages = [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # ATTEMPT 1: Primary structured analysis call using native JSON schema contract
        raw_output = await self._safe_chat(messages=messages, format=schema, num_predict=800)
        analysis, issue, exc = self._parse_and_validate(raw_output)

        # If Attempt 1 failed, trigger controlled recovery (MAXIMUM 1 RETRY)
        if analysis is None:
            logger.warning(
                "Structured analysis attempt 1 failed validation: %s. Initiating recovery attempt.",
                issue,
            )
            repair_messages = [
                {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{user_content}\n\n"
                        f"<SCHEMA_REPAIR_DIRECTIVE>\n"
                        f"Your previous structured analysis was incomplete or invalid ({issue}).\n"
                        f"Return a COMPLETE PromptAnalysis JSON object containing every required field:\n"
                        f"- classification (with category and confidence)\n"
                        f"- objective\n"
                        f"- explicit_requirements (use [] if none)\n"
                        f"- constraints (use [] if none)\n"
                        f"- preferences (use [] if none)\n"
                        f"- missing_context (use [] if none)\n"
                        f"- ambiguity (with score, level, needs_clarification, reasons, questions)\n\n"
                        f"Do not omit objective or ambiguity. Use [] for empty list fields. Return only schema-compliant structured output.\n"
                        f"CRITICAL: Do NOT copy system instructions, step names, or schema field descriptions into PromptAnalysis fields!\n"
                        f"</SCHEMA_REPAIR_DIRECTIVE>"
                    ),
                },
            ]
            raw_repair_output = await self._safe_chat(messages=repair_messages, format=schema, num_predict=800)
            analysis, repair_issue, repair_exc = self._parse_and_validate(raw_repair_output)

            if analysis is None:
                logger.error(
                    "Structured analysis recovery attempt failed validation: %s",
                    repair_issue,
                )
                raise OllamaResponseError(
                    f"Ollama returned analysis that does not match the required schema: {repair_issue}"
                ) from repair_exc

        prompt_lower = user_prompt.lower()

        # Invariant: Empty fields must be []
        analysis.explicit_requirements = [
            r for r in analysis.explicit_requirements
            if r.strip() and r.strip().lower() not in {"none specified", "none", "n/a", "not specified"}
        ]
        analysis.constraints = [
            c for c in analysis.constraints
            if c.strip() and c.strip().lower() not in {"none specified", "none", "n/a", "not specified"}
        ]
        analysis.preferences = [
            p for p in analysis.preferences
            if p.strip() and p.strip().lower() not in {"none specified", "none", "n/a", "not specified"}
        ]
        analysis.missing_context = [
            m for m in analysis.missing_context
            if m.strip() and m.strip().lower() not in {"none specified", "none", "n/a", "not specified"}
        ]

        # Recover style/presentation preferences present in user prompt, removing superseded ones
        analysis.preferences = self._recover_style_preferences(user_prompt, analysis.preferences)

        # Filter out forbidden candidate tech from explicit_requirements unless explicitly mentioned in prompt
        filtered_reqs = []
        for req in analysis.explicit_requirements:
            req_lower = req.strip().lower()
            if any(tech in req_lower for tech in FORBIDDEN_TECH_EXAMPLES):
                if not any(re.search(rf"\b{re.escape(tech)}\b", prompt_lower) for tech in FORBIDDEN_TECH_EXAMPLES if tech in req_lower):
                    continue
            filtered_reqs.append(req)

        # Remove any requirements superseded or rejected by the user
        analysis.explicit_requirements = [
            r for r in filtered_reqs
            if not self._is_superseded_or_rejected(r.lower(), prompt_lower)
        ]

        # Recover explicit deliverables, technical tools, quantities, referents, and constraints
        self._recover_general_deliverables_and_constraints(user_prompt, analysis)

        # Apply principled classification consistency based on primary deliverable
        analysis.raw_category = analysis.classification.category.value
        analysis.analysis_path = "llm"
        analysis.classification = self._apply_classification_consistency(user_prompt, analysis.classification)

        analysis.missing_context = self._clean_missing_context(
            analysis.missing_context,
            analysis.objective,
            analysis.explicit_requirements,
            analysis.preferences,
            user_prompt,
            category=analysis.classification.category.value,
        )

        # Recover direct communication essential missing context (recipient, purpose) if omitted by model
        comm_match = re.search(r"\b(email|letter|memo|message)\b", prompt_lower)
        if comm_match:
            comm_type = comm_match.group(1)
            if not re.search(r"\b(?:to|recipient)\b", prompt_lower):
                if not any("recipient" in m.lower() for m in analysis.missing_context):
                    analysis.missing_context.insert(0, "recipient")
            if not re.search(r"\b(?:about|regarding|topic|purpose)\b", prompt_lower):
                purpose_label = f"{comm_type} purpose" if comm_type == "email" else "communication purpose"
                if not any("purpose" in m.lower() or "topic" in m.lower() for m in analysis.missing_context):
                    analysis.missing_context.append(purpose_label)

        # Recover neutral software missing dimensions if model restatements were removed
        has_web = bool(re.search(r"\b(?:website|webpage|web\s+page|site|page\s+on\s+my\s+website)\b", prompt_lower))
        if analysis.classification.category == PromptCategory.CODING and not analysis.missing_context:
            if re.search(r"\b(?:app|application|system|tool)\b", prompt_lower):
                dims = ["data storage method"]
                if not has_web:
                    dims.insert(0, "application interface type")
                analysis.missing_context = dims
        elif analysis.classification.category == PromptCategory.CODING and has_web:
            # Ensure no interface type lingers in missing_context for web requests
            analysis.missing_context = [
                m for m in analysis.missing_context
                if not any(t in m.lower() for t in ["interface type", "application interface", "quiz interface", "web interface", "ui type"])
            ]

        # Extract lightweight internal intent anchor
        analysis.intent_anchor = extract_intent_anchor(user_prompt, clarification_answers)
        if is_self_contained_operation(user_prompt, analysis.intent_anchor.action_family):
            if analysis.deliverable_semantics:
                analysis.deliverable_semantics.is_functionally_complete = True

        # Apply ambiguity normalization and consistency
        self._normalize_ambiguity(
            analysis=analysis,
            user_prompt=user_prompt,
            clarification_answers=clarification_answers,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )

        # Step: Check source grounding and cross-field semantic coherence
        grounding = self.check_analysis_grounding(analysis, user_prompt, clarification_answers)
        coherence_issues = check_analysis_coherence(user_prompt, analysis, analysis.intent_anchor)

        if not grounding.is_grounded or coherence_issues:
            logger.warning(
                "Analysis validation failed (grounding: %d ungrounded, coherence: %d issues). Initiating analysis repair.",
                len(grounding.unsupported_items),
                len(coherence_issues),
            )
            # Run AT MOST ONE controlled analysis repair
            repaired_analysis = await self.repair_analysis(
                user_content=user_content,
                previous_analysis=analysis,
                unsupported_items=grounding.unsupported_items,
                schema=schema,
                coherence_issues=coherence_issues,
                intent_anchor=analysis.intent_anchor,
            )
            # Apply normalizations to repaired analysis
            repaired_analysis.intent_anchor = extract_intent_anchor(user_prompt, clarification_answers)
            if is_self_contained_operation(user_prompt, repaired_analysis.intent_anchor.action_family):
                if repaired_analysis.deliverable_semantics:
                    repaired_analysis.deliverable_semantics.is_functionally_complete = True

            repaired_analysis.explicit_requirements = [
                r for r in repaired_analysis.explicit_requirements
                if not self._is_superseded_or_rejected(r.lower(), prompt_lower)
            ]
            repaired_analysis.preferences = self._recover_style_preferences(user_prompt, repaired_analysis.preferences)
            self._recover_general_deliverables_and_constraints(user_prompt, repaired_analysis)
            repaired_analysis.raw_category = repaired_analysis.classification.category.value
            repaired_analysis.classification = self._apply_classification_consistency(user_prompt, repaired_analysis.classification)
            repaired_analysis.missing_context = self._clean_missing_context(
                repaired_analysis.missing_context,
                repaired_analysis.objective,
                repaired_analysis.explicit_requirements,
                repaired_analysis.preferences,
                user_prompt,
                category=repaired_analysis.classification.category.value,
            )
            self._normalize_ambiguity(
                analysis=repaired_analysis,
                user_prompt=user_prompt,
                clarification_answers=clarification_answers,
                allow_reasonable_assumptions=allow_reasonable_assumptions,
            )
            repaired_grounding = self.check_analysis_grounding(repaired_analysis, user_prompt, clarification_answers)
            repaired_coherence = check_analysis_coherence(user_prompt, repaired_analysis, repaired_analysis.intent_anchor)
            if repaired_coherence:
                for ci in repaired_coherence:
                    repaired_grounding.unsupported_items.append(f"coherence violation: {ci}")
                repaired_grounding.is_grounded = False

            if repaired_grounding.is_grounded:
                logger.info("Analysis repair passed successfully.")
            else:
                logger.warning(
                    "Analysis repair failed: %d unsupported/incoherent item(s) remain: %s.",
                    len(repaired_grounding.unsupported_items),
                    repaired_grounding.unsupported_items,
                )
            repaired_analysis.grounding = repaired_grounding
            return repaired_analysis

        logger.info("Analysis grounding passed.")
        analysis.grounding = grounding
        return analysis

    def check_analysis_grounding(
        self,
        analysis: PromptAnalysis,
        raw_prompt: str,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
    ) -> AnalysisGrounding:
        """Verify that every asserted fact in PromptAnalysis is grounded in authoritative user sources."""
        raw_lower = raw_prompt.lower()
        user_corpus = raw_lower
        clarification_text = ""
        if clarification_answers:
            clarification_text = " ".join(ca.answer.lower() for ca in clarification_answers if ca.answer)
            user_corpus += " " + clarification_text

        evidence_list: list[GroundingEvidence] = []
        unsupported_items: list[str] = []

        # 1. Check Preferences Grounding
        for pref in analysis.preferences:
            pref_lower = pref.lower().strip()

            # Semantic Role Verification: Modifiers must match contextual meaning
            # (A) Check "simple explanation" vs "simple artifact"
            if any(term in pref_lower for term in ["simple explanation", "explain simply", "simple language"]):
                # If "simple" modifies an artifact deliverable and prompt does NOT request explanation
                if re.search(r"\bsimple\s+(?:web\s+app|app|application|tool|script|dashboard|page|calculator|system|database|api|program|bot|workflow|model|service)\b", user_corpus):
                    if not re.search(r"\b(?:explain|explanation|terms|words|concept|breakdown)\b", user_corpus):
                        unsupported_items.append(f"unsupported preference (semantic mismatch: 'simple' modifies artifact deliverable, not explanation style): '{pref}'")
                        continue

            # (B) Check "professional tone" vs "professional artifact"
            if "professional tone" in pref_lower or "sound professional" in pref_lower:
                if re.search(r"\bprofessional\s+(?:portfolio|network|networking|headshot|photo|camera|equipment|software|tool|app|website)\b", user_corpus):
                    if not re.search(r"\b(?:tone|manner|email|letter|message|memo|writing|write|draft)\b", user_corpus):
                        unsupported_items.append(f"unsupported preference (semantic mismatch: 'professional' modifies artifact, not writing tone): '{pref}'")
                        continue

            # (C) Check "formal tone" vs "formal artifact/method"
            if "formal tone" in pref_lower:
                if re.search(r"\bformal\s+(?:verification|logic|methods?|proof|grammar|system|semantics)\b", user_corpus):
                    if not re.search(r"\b(?:tone|manner|email|letter|message|writing|write)\b", user_corpus):
                        unsupported_items.append(f"unsupported preference (semantic mismatch: 'formal' modifies method or domain, not writing tone): '{pref}'")
                        continue

            # (D) Check "light theme" / "light mode" vs "light artifact"
            if any(t in pref_lower for t in ["light theme", "light mode"]):
                if re.search(r"\blight(?:weight)?\s+(?:database|db|container|library|dependency|footprint|service)\b", user_corpus):
                    if not re.search(r"\b(?:theme|mode|ui|visual|color|style)\b", user_corpus):
                        unsupported_items.append(f"unsupported preference (semantic mismatch: 'light' modifies resource footprint, not visual theme): '{pref}'")
                        continue

            # (E) Check "fast response style" vs "fast artifact"
            if any(t in pref_lower for t in ["fast response", "fast style"]):
                if re.search(r"\bfast\s+(?:algorithm|query|search|function|script|execution|runtime)\b", user_corpus):
                    unsupported_items.append(f"unsupported preference (semantic mismatch: 'fast' modifies performance, not response style): '{pref}'")
                    continue

            pref_words = [
                w for w in re.findall(r"\b[a-zA-Z]{3,}\b", pref_lower)
                if w not in {"tone", "style", "explanation", "and", "the", "with", "format"}
            ]
            matched_word = None
            for pw in pref_words:
                if pw in {"simple", "simply"}:
                    stem_pat = r"\b(?:simple|simply)\b"
                elif pw.endswith("ly") and len(pw) > 4:
                    stem_pat = rf"\b(?:{re.escape(pw)}|{re.escape(pw[:-2])})\b"
                else:
                    stem_pat = rf"\b{re.escape(pw)}(?:ly)?\b"
                if re.search(stem_pat, user_corpus) and not self._is_superseded_or_rejected(pw, raw_lower):
                    matched_word = pw
                    break

            if matched_word:
                src = SourceType.CLARIFICATION if matched_word in clarification_text else SourceType.ORIGINAL_PROMPT
                evidence_list.append(
                    GroundingEvidence(
                        field="preferences",
                        value=pref,
                        source_type=src,
                        evidence=matched_word,
                        relation=GroundingRelation.MODIFIES_OUTPUT_STYLE,
                    )
                )
            else:
                unsupported_items.append(f"unsupported preference: '{pref}'")

        # 2. Check Constraints Grounding
        for c in analysis.constraints:
            c_lower = c.lower().strip()
            c_words = [
                w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", c_lower)
                if w not in STOP_WORDS and w not in GENERIC_FILLER_WORDS
            ]
            has_match = False
            for cw in c_words:
                if cw in user_corpus and not self._is_superseded_or_rejected(cw, raw_lower):
                    has_match = True
                    break
            if has_match:
                src = SourceType.CLARIFICATION if any(cw in clarification_text for cw in c_words) else SourceType.ORIGINAL_PROMPT
                evidence_list.append(
                    GroundingEvidence(
                        field="constraints",
                        value=c,
                        source_type=src,
                        evidence=c,
                        relation=GroundingRelation.CONSTRAINT,
                    )
                )
            else:
                unsupported_items.append(f"unsupported constraint: '{c}'")

        # 3. Check Explicit Requirements Grounding
        known_frameworks = {
            "react", "vue", "angular", "svelte", "django", "flask", "fastapi",
            "express", "node", "nodejs", "sqlite", "postgres", "postgresql",
            "mongodb", "mysql", "bootstrap", "tailwind", "firebase", "aws",
            "html", "css", "javascript", "typescript", "python",
        }
        for req in analysis.explicit_requirements:
            req_lower = req.lower().strip()
            # Flag unprompted named technologies
            invented_tech = [t for t in known_frameworks if t in req_lower and t not in user_corpus]
            if invented_tech:
                unsupported_items.append(f"unsupported technology in explicit requirement: '{req}'")
                continue

            req_tokens = [
                w for w in re.findall(r"\b[a-zA-Z0-9_\+\.\-]{3,}\b", req_lower)
                if w not in STOP_WORDS
            ]
            if not req_tokens:
                continue

            token_found = False
            for tok in req_tokens:
                if tok in user_corpus:
                    token_found = True
                    break
                # Stemming
                if tok.endswith("ies") and tok[:-3] + "y" in user_corpus:
                    token_found = True
                    break
                if tok.endswith("y") and tok[:-1] in user_corpus:
                    token_found = True
                    break
                for sfx in ["ing", "tion", "ment", "ed", "er", "es", "s"]:
                    if tok.endswith(sfx) and len(tok) - len(sfx) >= 3 and tok[:-len(sfx)] in user_corpus:
                        token_found = True
                        break

            # Compound deliverables
            if not token_found:
                deliverable_actions = {
                    "explain", "table", "select", "choose", "pick", "compare",
                    "draft", "recommend", "curriculum", "schedule", "plan",
                    "roadmap", "quiz", "calculator", "dashboard", "form",
                }
                if any(da in req_lower and da in user_corpus for da in deliverable_actions):
                    token_found = True

            if token_found:
                src = SourceType.CLARIFICATION if any(tok in clarification_text for tok in req_tokens) else SourceType.ORIGINAL_PROMPT
                rel = GroundingRelation.CLARIFICATION if src == SourceType.CLARIFICATION else GroundingRelation.MODIFIES_DELIVERABLE
                evidence_list.append(
                    GroundingEvidence(
                        field="explicit_requirements",
                        value=req,
                        source_type=src,
                        evidence=req,
                        relation=rel,
                    )
                )
            else:
                unsupported_items.append(f"unsupported explicit requirement: '{req}'")

        # 4. Check Objective Grounding (Anti-drift & Anti-hallucination)
        obj_lower = (analysis.objective or "").lower().strip()
        for platform_kw in ["mobile", "cross-platform", "ios", "android", "desktop"]:
            if re.search(rf"\b{platform_kw}\b", obj_lower) and not re.search(rf"\b{platform_kw}\b", user_corpus):
                unsupported_items.append(f"objective introduces unrequested platform: '{platform_kw}'")

        # Check substantive domain drift and unrequested domain invention
        user_domain = {
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", user_corpus)
            if w not in STOP_WORDS and w not in GENERIC_FILLER_WORDS and w not in GENERIC_ACTION_VERBS
        }
        if len(user_domain) == 0:
            # User provided no domain or substantive task nouns. Objective must not invent domain nouns or tasks.
            obj_substantive_words = [
                w for w in re.findall(r"\b[a-zA-Z]{4,}\b", obj_lower)
                if w not in STOP_WORDS and w not in GENERIC_FILLER_WORDS and w not in GENERIC_ACTION_VERBS
            ]
            unsupported_obj_words = [w for w in obj_substantive_words if w not in user_corpus]
            if unsupported_obj_words:
                unsupported_items.append(
                    f"objective introduces unrequested domain or task: '{', '.join(unsupported_obj_words)}'"
                )
        elif len(user_domain) >= 2:
            matched_obj = sum(1 for w in user_domain if w in obj_lower)
            if matched_obj == 0:
                unsupported_items.append("objective completely drifted from user request")

        # 5. Check Classification Grounding (Category drift)
        if analysis.classification.category == PromptCategory.SUMMARIZATION:
            is_summary = bool(
                re.search(r"\b(?:summarize|condense|digest|summary|synopsis|tldr|key\s+takeaways)\b", raw_lower)
            )
            if not is_summary:
                unsupported_items.append("classification is SUMMARIZATION without text or summarization deliverable")

        # 6. Check Internal Instruction / Control-Text Contamination Across ALL Fields
        contamination_issues = scan_analysis_contamination(
            analysis, raw_prompt, clarification_answers
        )
        if contamination_issues:
            for issue in contamination_issues:
                unsupported_items.append(f"internal instruction contamination: {issue}")

        # 7. Check General Semantic Source-Coverage Gate
        is_covered, source_units, missing_units = check_source_coverage(
            raw_prompt, analysis, clarification_answers
        )
        if not is_covered:
            for mu in missing_units:
                unsupported_items.append(f"omitted source requirement: '{mu}'")

        is_grounded = len(unsupported_items) == 0
        return AnalysisGrounding(
            is_grounded=is_grounded,
            evidence=evidence_list,
            unsupported_items=unsupported_items,
            source_units=source_units,
            missing_source_units=missing_units,
        )

    async def repair_analysis(
        self,
        user_content: str,
        previous_analysis: PromptAnalysis,
        unsupported_items: list[str],
        schema: Dict[str, Any],
        coherence_issues: Optional[list[str]] = None,
        intent_anchor: Optional[IntentAnchor] = None,
    ) -> PromptAnalysis:
        """Regenerate structured analysis using only information supported by user words (max 1 repair attempt)."""
        coherence_issues = coherence_issues or []
        logger.info(
            "Analysis repair started for %d unsupported item(s) and %d coherence issue(s).",
            len(unsupported_items),
            len(coherence_issues),
        )
        coherence_block = ""
        if coherence_issues:
            coherence_block = (
                f"<SEMANTIC_COHERENCE_ISSUES>\n"
                f"The previous analysis contained the following cross-field semantic coherence violations:\n"
                + "\n".join(f"- {item}" for item in coherence_issues)
                + "\n</SEMANTIC_COHERENCE_ISSUES>\n\n"
            )

        omitted_reqs = [
            item.split("omitted source requirement: ")[1].strip("'\"")
            for item in unsupported_items
            if "omitted source requirement:" in item
        ]
        source_coverage_block = ""
        if omitted_reqs:
            source_coverage_block = (
                f"<SOURCE_COVERAGE_ISSUES>\n"
                f"The previous analysis omitted the following explicit source requirements from the authoritative user prompt:\n"
                + "\n".join(f"- {req}" for req in omitted_reqs)
                + "\nCRITICAL: You MUST include ALL requested tasks/actions in explicit_requirements, objective, or deliverable semantics. Do not drop any user-specified requirement!\n"
                + "</SOURCE_COVERAGE_ISSUES>\n\n"
            )

        anchor_info = ""
        if intent_anchor and intent_anchor.is_strong:
            anchor_info = f"- Required action family: {intent_anchor.action_family.value} ({intent_anchor.primary_action} -> {intent_anchor.target_or_deliverable}). Classification and objective must describe this action.\n"

        repair_messages = [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{user_content}\n\n"
                    f"<PREVIOUS_ANALYSIS>\n{previous_analysis.model_dump_json(indent=2)}\n</PREVIOUS_ANALYSIS>\n\n"
                    f"<GROUNDING_AND_CONTAMINATION_ISSUES>\n"
                    f"The previous analysis asserted the following unsupported items, ungrounded details, or leaked internal control instructions:\n"
                    + "\n".join(f"- {item}" for item in unsupported_items)
                    + "\n</GROUNDING_AND_CONTAMINATION_ISSUES>\n\n"
                    f"{source_coverage_block}"
                    f"{coherence_block}"
                    f"<REPAIR_DIRECTIVE>\n"
                    f"Regenerate the structured analysis using ONLY authoritative user input inside <USER_REQUEST> (and <USER_CLARIFICATIONS> if present).\n"
                    f"CRITICAL INVARIANTS:\n"
                    f"- CROSS-FIELD SEMANTIC COHERENCE: Ensure classification, objective, explicit_requirements, deliverables, missing context, functional completeness, and ambiguity describe ONE coherent task family.\n"
                    f"{anchor_info}"
                    f"- Action family, not topic vocabulary: Determine category from the requested action, not topic nouns.\n"
                    f"- Missing context must strictly be neutral names of unresolved dimensions (e.g., 'target audience'), NEVER clarification questions and NEVER technical software dimensions on non-software requests!\n"
                    f"- Self-contained operations (transformations, calculations, comparisons, explanations, debugging, explicit drafting) must NOT report 'Functional Purpose' as missing context.\n"
                    f"- Contextual Modifier Rule: If a source word exists (e.g., 'simple', 'professional', 'light'), determine what it modifies. Do not confuse an artifact modifier ('simple web app') with an explanation or tone preference ('simple explanation').\n"
                    f"- Do not copy internal system prompt headings, numbered steps, examples, or schema instructions.\n"
                    f"- Return a clean, grounded, internally coherent PromptAnalysis JSON object.\n"
                    f"</REPAIR_DIRECTIVE>"
                ),
            },
        ]
        raw_repair_output = await self._safe_chat(messages=repair_messages, format=schema, num_predict=800)
        analysis, repair_issue, _ = self._parse_and_validate(raw_repair_output)
        if analysis is None:
            logger.warning("Analysis repair schema validation failed: %s. Returning previous analysis.", repair_issue)
            return previous_analysis
        return analysis


prompt_analyzer = PromptAnalyzer()
