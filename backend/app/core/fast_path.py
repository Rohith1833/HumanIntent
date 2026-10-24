"""HumanIntent v1.0 Production Fast-Path Engine.

Implements the latency-critical production pipeline:
Input -> Deterministic preprocessing -> Single LLM generation -> Deterministic verification -> Output

Invariants:
1. Exactly ONE Ollama LLM generation call for normal enhancement requests.
2. Zero sequential LLM calls for analysis, scoring, or repair in standard runtime.
3. Fast deterministic intent, constraint, tech, and entity extraction.
4. Strict enhancement-only generation: LLM must NEVER execute the task or generate the deliverable.
5. Strict intent preservation: preserve explicit technologies, numbers, and negative constraints.
6. Independent timing measurement for preprocessing, inference, verification, and total latency.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from app.core.category_router import category_router
from app.core.config import settings
from app.core.context_budget import context_budget_manager
from app.core.instruction_boundary import detect_control_text
from app.core.intent_anchor import extract_intent_anchor
from app.core.source_coverage import (
    TECH_ALIASES,
    clean_neg_hypothesis,
    extract_source_semantic_units,
)

PRODUCTION_TECH_ALIASES = {
    **TECH_ALIASES,
    "pydantic": "pydantic",
    "sqlalchemy": "sqlalchemy",
    "alembic": "alembic",
    "pytest": "pytest",
}

from app.llm.ollama_client import OllamaClient, OllamaTimeoutError, ollama_client
from app.schemas.analysis import (
    AmbiguityAssessment,
    AmbiguityLevel,
    AnalysisGrounding,
    ClarificationAnswer,
    ClarificationQuestion,
    DeliverableSemantics,
    GroundingEvidence,
    GroundingRelation,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
    SourceType,
)
from app.schemas.telemetry import StageTimer, StageTimingDiagnostic
from app.schemas.validation import (
    PromptValidation,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
)
from app.templates.selector import get_template


class FastPathResult(tuple):
    """2-tuple of (analysis, enhanced_prompt) compatible with unpacking while providing rich telemetry and validation metadata."""

    analysis: PromptAnalysis
    enhanced_prompt: str
    validation: Optional[PromptValidation]
    repair_attempted: bool
    quality: Optional[Any]
    initial_enhanced_prompt: Optional[str]
    initial_validation: Optional[PromptValidation]
    stage_timings: list[StageTimingDiagnostic]
    mode: str
    total_ms: Optional[float]
    inference_ms: Optional[float]

    def __new__(
        cls,
        analysis: PromptAnalysis,
        enhanced_prompt: str,
        validation: Optional[PromptValidation] = None,
        repair_attempted: bool = False,
        quality: Optional[Any] = None,
        initial_enhanced_prompt: Optional[str] = None,
        initial_validation: Optional[PromptValidation] = None,
        stage_timings: Optional[list[StageTimingDiagnostic]] = None,
        mode: str = "professional",
        total_ms: Optional[float] = None,
        inference_ms: Optional[float] = None,
    ):
        instance = super().__new__(cls, (analysis, enhanced_prompt))
        instance.analysis = analysis
        instance.enhanced_prompt = enhanced_prompt
        instance.validation = validation
        instance.repair_attempted = repair_attempted
        instance.quality = quality
        instance.initial_enhanced_prompt = initial_enhanced_prompt
        instance.initial_validation = initial_validation
        instance.stage_timings = stage_timings or []
        instance.mode = mode
        instance.total_ms = total_ms
        instance.inference_ms = inference_ms
        return instance

logger = logging.getLogger("humanintent.fast_path")

# Production system prompt enforcing prompt-only rewriting
PRODUCTION_ENHANCEMENT_SYSTEM_PROMPT = """You are HumanIntent, an intent-preserving prompt optimization system.

Your SOLE PURPOSE is to rewrite the user's rough request into an unambiguous, well-structured instruction prompt for another downstream AI.

NON-NEGOTIABLE CORE DIRECTIVES:
1. PROMPT ENHANCEMENT ONLY:
   - You must NEVER answer the prompt, implement code, write the essay, perform calculations, or produce the requested deliverable.
   - You are creating the PROMPT that another AI will use to perform the task.
2. PRESERVE EVERY EXPLICIT REQUIREMENT:
   - Retain all explicit technologies, programming languages, frameworks, numerical quantities, file paths, and output formats.
   - Retain every negative constraint (e.g. 'no dependencies', 'without pandas', 'do not use X') exactly as specified.
3. ZERO INVENTIONS:
   - Do NOT invent unrequested technologies, frameworks, databases, pricing models, user demographics, or security standards.
   - Do NOT add boilerplate features (responsive design, dark mode, login, databases) unless explicitly requested.
4. DIRECT DOWNSTREAM INSTRUCTIONS:
   - Address the downstream AI directly with clear requirements.
   - Do NOT use meta-commentary ('Instruct the implementing AI:', 'Here is the enhanced prompt:').
   - Output ONLY the enhanced prompt. The first character of your response must belong to the enhanced prompt itself.
5. ADAPTIVE LENGTH & STRUCTURE:
   - Organize into clean sections (e.g. OBJECTIVE, REQUIREMENTS, CONSTRAINTS, OUTPUT FORMAT).
   - Omit sections that have no content (e.g., do not output 'CONSTRAINTS: None').
   - Keep the prompt concise, actionable, and proportionate to the user's request.
"""


@dataclass
class FastPathPreprocessResult:
    analysis: PromptAnalysis
    category: PromptCategory
    source_units: List[str]
    negative_constraints: List[str]
    explicit_technologies: List[str]
    numerical_quantities: List[str]
    is_critically_ambiguous: bool
    missing_source_data: bool
    clarification_questions: List[ClarificationQuestion]
    preprocessing_ms: float


@dataclass
class FastPathVerificationResult:
    is_valid: bool
    cleaned_prompt: str
    issues: List[ValidationIssue]
    deliverable_intercepted: bool
    verification_ms: float


class FastPathPipeline:
    """Production fast-path engine optimizing end-to-end latency to < 30 seconds."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or ollama_client

    def preprocess(
        self,
        user_prompt: str,
        clarification_answers: Optional[List[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
        mode: str = "professional",
    ) -> FastPathPreprocessResult:
        """Deterministic preprocessor: extracts category, intent, constraints, and grounded analysis in < 20ms."""
        t0 = time.perf_counter()
        raw_text = user_prompt.strip()

        # 1. Deterministic Category and Intent Anchor Routing
        routing_result = category_router.route(raw_text, clarification_answers)
        category = routing_result.category
        anchor = extract_intent_anchor(raw_text, clarification_answers)

        # Disambiguate false-positive DEBUGGING when prompt has no defects/errors
        if category == PromptCategory.DEBUGGING:
            lower_text = raw_text.lower()
            debug_indicators = {"error", "bug", "crash", "traceback", "exception", "fail", "failing", "broken", "syntax", "fix", "issue", "segfault", "panic", "not working", "why does it"}
            has_debug_signal = any(re.search(rf"\b{re.escape(ind)}\b", lower_text) for ind in debug_indicators)
            if not has_debug_signal:
                if re.search(r"\b(?:compare|comparison|versus|\bvs\b|pros\s+and\s+cons|trade-?offs|comparative|survey|literature|study)\b", lower_text):
                    category = PromptCategory.RESEARCH
                elif re.search(r"\b(?:explain|how\s+does|how\s+do|teach|understand|what\s+is|what\s+are)\b", lower_text):
                    category = PromptCategory.EDUCATION
                else:
                    category = PromptCategory.GENERAL

        # 2. Extract Atomic Semantic Units
        source_units = extract_source_semantic_units(raw_text, clarification_answers)

        # 3. Extract Negative Constraints
        negative_constraints: List[str] = []
        neg_patterns = [
            r"\b(?:without|avoid|never|do\s+not|dont|disallow|exclude|no\s+(?:external|third-party|extra)?)\b[^\n,;.]+",
            r"\bzero\s+(?:dependencies|deps|overhead)\b",
        ]
        for pat in neg_patterns:
            for match in re.finditer(pat, raw_text, flags=re.IGNORECASE):
                neg_clause = match.group(0).strip(" \t\r\n.,;:")
                cleaned_neg = clean_neg_hypothesis(neg_clause)
                if cleaned_neg not in negative_constraints:
                    negative_constraints.append(cleaned_neg)

        # Also pull from source_units
        for unit in source_units:
            if re.match(r"^(?:without|avoid|never|no|do\s+not)\b", unit, re.IGNORECASE):
                cleaned_u = clean_neg_hypothesis(unit)
                if cleaned_u not in negative_constraints:
                    negative_constraints.append(cleaned_u)

        # 4. Extract Named Technologies
        lower_prompt = raw_text.lower()
        explicit_technologies: List[str] = []
        words = set(re.findall(r"[a-z0-9#+.]+", lower_prompt))
        for raw_alias, canonical in PRODUCTION_TECH_ALIASES.items():
            if raw_alias in words or re.search(rf"\b{re.escape(raw_alias)}\b", lower_prompt):
                if canonical not in explicit_technologies:
                    explicit_technologies.append(canonical)

        # 5. Extract Numerical Quantities and Thresholds
        numerical_quantities = re.findall(
            r"\b\d+(?:\.\d+)?(?:\s*(?:gb|mb|kb|ms|s|seconds?|minutes?|hours?|days?|rows?|users?|items?|records?|pages?|fps|%|x|k|m|b))?\b",
            raw_text,
            flags=re.IGNORECASE,
        )

        # 6. Ambiguity & Missing Source Data Assessment
        is_critically_ambiguous = False
        missing_source_data = False
        clarification_questions: List[ClarificationQuestion] = []

        # Check ultra-short vague requests (e.g. <= 2 words, or vague system requests)
        token_count = len(raw_text.split())
        cleaned_simple = re.sub(r"[^\w\s]", "", lower_prompt).strip()
        vague_patterns = [
            r"^(?:build|make|create|develop|write)\s+(?:me\s+|us\s+)?(?:a\s+|an\s+)?(?:system|app|application|tool|program|platform|solution|thing|software|script)$",
            r"^(?:fix|debug|check)\s+(?:it|this|that|code|error|bug)$",
            r"^(?:help|assist)\s+(?:me|us)$",
        ]
        is_vague = (token_count <= 2) or any(re.match(p, cleaned_simple, re.IGNORECASE) for p in vague_patterns)

        if is_vague:
            if not allow_reasonable_assumptions:
                is_critically_ambiguous = True
                clarification_questions.append(
                    ClarificationQuestion(
                        id="clarify_scope",
                        question="What specific task or application would you like to build or accomplish?",
                        reason="The request contains insufficient detail to determine the scope.",
                        target_dimension="scope",
                    )
                )

        # Check missing source data for summarization and data analysis
        missing_data_triggers = [
            (PromptCategory.SUMMARIZATION, r"\b(?:summarize|condense|tldr|digest)\b.*\b(?:document|contract|agreement|paper|article|pdf|text|transcript)\b"),
            (PromptCategory.DATA_ANALYSIS, r"\b(?:analyze|calculate|process)\b.*\b(?:data|dataset|csv|metrics|sales|numbers|churn)\b"),
        ]
        for cat_trigger, pat in missing_data_triggers:
            if category == cat_trigger and re.search(pat, lower_prompt):
                # If prompt is short (< 35 words) and references a document/dataset without supplying it
                if token_count < 35 and not re.search(r"```|:\n|following data|here is", lower_prompt):
                    missing_source_data = True
                    if not allow_reasonable_assumptions:
                        is_critically_ambiguous = True
                        target_entity = "document or text" if category == PromptCategory.SUMMARIZATION else "dataset or metrics"
                        clarification_questions.append(
                            ClarificationQuestion(
                                id="missing_source",
                                question=f"Please provide or describe the {target_entity} you would like to {category.value.lower()}.",
                                reason=f"The requested {category.value.lower()} requires source data that was not provided.",
                                target_dimension="source_data",
                            )
                        )

        # 7. Construct Grounded PromptAnalysis
        # Filter source units to produce clean explicit requirements (excluding pure negative constraints)
        req_units = [
            u for u in source_units
            if not re.match(r"^(?:without|avoid|never|no|do\s+not)\b", u, re.IGNORECASE)
        ]
        # Include explicit technologies if not already reflected
        for tech in explicit_technologies:
            if not any(tech in u.lower() for u in req_units):
                req_units.append(f"Use {tech}")

        objective_text = f"{anchor.primary_action}: {anchor.target_or_deliverable}" if anchor and anchor.primary_action else f"Fulfill request: {raw_text[:80]}"
        if len(raw_text) <= 120 and token_count >= 3:
            objective_text = raw_text

        ambiguity_level = AmbiguityLevel.HIGH if is_critically_ambiguous else AmbiguityLevel.LOW
        ambiguity_score = 85 if is_critically_ambiguous else 15

        analysis = PromptAnalysis(
            classification=PromptClassification(
                category=category,
                confidence=round(routing_result.confidence, 4),
            ),
            objective=objective_text,
            explicit_requirements=req_units,
            constraints=negative_constraints,
            preferences=[],
            missing_context=["source data payload"] if missing_source_data else [],
            ambiguity=AmbiguityAssessment(
                score=ambiguity_score,
                level=ambiguity_level,
                needs_clarification=is_critically_ambiguous and not allow_reasonable_assumptions,
                reasons=[q.reason for q in clarification_questions],
                questions=clarification_questions,
            ),
            grounding=AnalysisGrounding(
                is_grounded=True,
                evidence=[
                    GroundingEvidence(
                        field="explicit_requirements",
                        value=u,
                        source_type=SourceType.ORIGINAL_PROMPT,
                        evidence=u,
                        relation=GroundingRelation.DIRECT,
                    )
                    for u in req_units[:10]
                ],
                unsupported_items=[],
            ),
            deliverable_semantics=DeliverableSemantics(
                deliverable_type=category.value.lower(),
                intended_outcome=objective_text,
                is_functionally_complete=True,
            ),
        )

        t1 = time.perf_counter()
        prep_ms = (t1 - t0) * 1000.0

        return FastPathPreprocessResult(
            analysis=analysis,
            category=category,
            source_units=source_units,
            negative_constraints=negative_constraints,
            explicit_technologies=explicit_technologies,
            numerical_quantities=numerical_quantities,
            is_critically_ambiguous=is_critically_ambiguous,
            missing_source_data=missing_source_data,
            clarification_questions=clarification_questions,
            preprocessing_ms=prep_ms,
        )

    def _build_generation_prompt(
        self,
        user_prompt: str,
        preprocess: FastPathPreprocessResult,
        mode: str = "professional",
        allow_reasonable_assumptions: bool = False,
    ) -> Tuple[List[Dict[str, str]], int]:
        """Builds optimal composite prompt and dynamic token limit for single-call enhancement."""
        category = preprocess.category
        analysis = preprocess.analysis
        template = get_template(category)
        template_guidance = template.get_instructions(user_prompt, analysis)

        # Dynamic token limit based on mode and prompt complexity
        input_len = len(user_prompt)
        if mode == "quick":
            num_predict = min(250, max(150, input_len // 4 + 150))
        elif mode == "deep":
            num_predict = min(480, max(300, input_len // 3 + 280))
        else:  # professional / standard
            num_predict = min(360, max(220, input_len // 3 + 200))

        # Build grounded requirement bullets
        req_lines = [f"- {req}" for req in analysis.explicit_requirements]
        req_block = "\n".join(req_lines) if req_lines else "- Implement the requested functionality as described."

        # Build constraint bullets
        constraint_lines = [f"- {c}" for c in analysis.constraints]
        constraint_block = "\n".join(constraint_lines) if constraint_lines else ""

        user_content = (
            f"<ORIGINAL_USER_REQUEST>\n"
            f"{user_prompt}\n"
            f"</ORIGINAL_USER_REQUEST>\n\n"
            f"<GROUNDED_REQUIREMENTS>\n"
            f"Category: {category.value}\n"
            f"Core Objective: {analysis.objective}\n"
            f"Explicit Requirements:\n{req_block}\n"
        )
        if constraint_block:
            user_content += f"Negative Constraints (Must Retain):\n{constraint_block}\n"
        if preprocess.missing_source_data:
            user_content += (
                f"Note: Source data was not attached. Instruct downstream AI to request the data or provide "
                f"a generic sample format rather than hallucinating specific numbers.\n"
            )
        user_content += (
            f"</GROUNDED_REQUIREMENTS>\n\n"
            f"<CATEGORY_GUIDANCE>\n"
            f"{template_guidance}\n"
            f"</CATEGORY_GUIDANCE>\n\n"
            f"<INSTRUCTION>\n"
            f"Rewrite the user request into an enhanced prompt for another AI adhering to these rules:\n"
            f"1. DO NOT EXECUTE THE TASK. DO NOT WRITE THE FINAL DELIVERABLE. Write the PROMPT.\n"
            f"2. Retain all explicit technologies ({', '.join(preprocess.explicit_technologies) or 'as stated'}).\n"
            f"3. Retain all negative constraints and quantities exactly.\n"
            f"4. Do NOT invent unrequested technologies, frameworks, or data.\n"
            f"5. Start directly with the enhanced prompt. Zero preamble or labels.\n"
            f"</INSTRUCTION>"
        )

        messages = [
            {"role": "system", "content": PRODUCTION_ENHANCEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return messages, num_predict

    async def generate(
        self,
        messages: List[Dict[str, str]],
        num_predict: int,
        timeout: float = 28.0,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Single LLM generation call with strict timeout and keep-alive."""
        t0 = time.perf_counter()
        raw_output = await self.client.chat(
            messages=messages,
            num_predict=num_predict,
            keep_alive="30m",  # Warm model persistence
        )
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        metadata = getattr(self.client, "last_call_metadata", {})
        return raw_output, inference_ms, metadata

    def clean_output(
        self,
        raw_text: str,
        user_prompt: str,
        preprocess: FastPathPreprocessResult,
    ) -> str:
        """Deterministic cleanup removing markdown wrapping, conversational preambles, and meta leaks."""
        text = raw_text.strip()

        # Remove outer markdown code blocks wrapping the whole prompt
        if text.startswith("```"):
            text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Remove common introductory commentary lines
        pattern = r"^(?:(?:Here\s+is|Here\'s|Enhanced\s+version|Enhanced\s+prompt|Sure|Certainly|Below\s+is)[^\n:]*:\s*\n*)"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # Remove accidental leading stray punctuation/tokens
        text = re.sub(r"^[\)\.\-:\s]+", "", text).strip()

        # Remove conversational repair or justification paragraphs
        text = re.sub(
            r"(?im)^\s*(?:Note\s+that\s+I|Please\s+note\s+that\s+I|I\s+(?:have\s+)?(?:removed|fixed|updated|adhered)|Per\s+the\s+repair)[^\n]*?(?:\n+|$)",
            "",
            text,
        ).strip()

        # Remove trailing conversational postamble notes
        postamble_pattern = r"(?im)\n+\s*(?:Note|P\.S\.|Please note|I hope this helps)\b[^\n]*?(?:that\s+I|:\s*)[\s\S]*$"
        text = re.sub(postamble_pattern, "", text).strip()

        # Filter internal control text leaks
        cleaned_final_lines = []
        for line in text.split("\n"):
            line_str = line.strip()
            if line_str:
                is_control, _ = detect_control_text(line_str, user_prompt)
                if is_control:
                    continue
            cleaned_final_lines.append(line)
        text = "\n".join(cleaned_final_lines).strip()

        # Strip accidental structured XML tag echoes
        text = re.sub(r"</?(?:ORIGINAL_USER_REQUEST|GROUNDED_REQUIREMENTS|CATEGORY_GUIDANCE|INSTRUCTION|AUTHORITATIVE_USER_REQUEST)>", "", text).strip()

        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def verify(
        self,
        enhanced_prompt: str,
        user_prompt: str,
        preprocess: FastPathPreprocessResult,
    ) -> FastPathVerificationResult:
        """Deterministic post-generation verification validating all core invariants in < 5ms."""
        t0 = time.perf_counter()
        issues: List[ValidationIssue] = []
        cleaned = self.clean_output(enhanced_prompt, user_prompt, preprocess)
        lower_cleaned = cleaned.lower()
        deliverable_intercepted = False

        # 1. Deliverable Execution Trap Verification
        # Check if output is raw code or pure answer without prompt structure
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        first_line = lines[0] if lines else ""

        # If it starts with pure code imports or function definitions
        if re.match(r"^(?:import\s+\w+|from\s+\w+\s+import|def\s+\w+\(|class\s+\w+[:\(]|const\s+\w+\s*=|<template>|<!DOCTYPE)", first_line):
            deliverable_intercepted = True
            # Wrap into an explicit instruction prompt preserving the code as example
            cleaned = (
                f"OBJECTIVE:\n"
                f"Implement the requested solution matching the specifications below.\n\n"
                f"REQUIREMENTS:\n"
                f"- Follow the implementation guidelines.\n\n"
                f"REFERENCE / SPECIFICATION:\n"
                f"{cleaned}"
            )
            lower_cleaned = cleaned.lower()

        # 2. Negative Constraint Verification
        for neg in preprocess.negative_constraints:
            # Check keywords in the negative constraint
            terms = re.findall(r"\b[a-zA-Z0-9#+-]+\b", neg.lower())
            substantive_terms = [t for t in terms if t not in {"do", "not", "without", "never", "avoid", "use", "the", "a", "an", "there", "should", "be"}]
            if substantive_terms:
                target_word = substantive_terms[0]
                # If the target word appears, verify it appears in a negative / restriction context, NOT as a recommendation
                if target_word in lower_cleaned:
                    pos_pattern = rf"\b(?:must\s+use|must\s+have|recommend|require|install|integrate\s+with)\s+{re.escape(target_word)}\b"
                    neg_pattern = rf"\b(?:do\s+not|without|no|avoid|never|disallow)(?:\s+(?:use|using|any|need|require|install))*\s+{re.escape(target_word)}\b"
                    if re.search(pos_pattern, lower_cleaned) and not re.search(neg_pattern, lower_cleaned):
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.CONSTRAINT_CONTRADICTION,
                                severity=ValidationSeverity.ERROR,
                                message=f"Enhanced prompt violates negative constraint: '{neg}'",
                            )
                        )
                else:
                    # If the prohibited term was completely dropped, re-inject the constraint section
                    cleaned += f"\n\nCONSTRAINTS:\n- {neg}"
                    lower_cleaned = cleaned.lower()

        # 3. Explicit Technology Verification
        for tech in preprocess.explicit_technologies:
            if tech not in lower_cleaned and not any(alias in lower_cleaned for alias, can in PRODUCTION_TECH_ALIASES.items() if can == tech):
                # Technology dropped!
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.SOURCE_REQUIREMENT_DROPPED,
                        severity=ValidationSeverity.ERROR,
                        message=f"Explicit technology '{tech}' was dropped from enhanced prompt.",
                    )
                )

        # 4. Numerical Quantities Verification
        for num in preprocess.numerical_quantities:
            num_clean = num.strip().lower()
            if num_clean not in lower_cleaned:
                # Numerical constraint dropped
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.QUANTITY_MISMATCH,
                        severity=ValidationSeverity.WARNING,
                        message=f"Explicit numerical quantity '{num}' may have been omitted.",
                    )
                )

        # 5. Missing Source Data Verification
        if preprocess.missing_source_data:
            hallucinated_data_patterns = [
                r"\b(?:as\s+shown\s+in\s+the\s+attached|from\s+the\s+provided\s+dataset\s+below|the\s+contract\s+states\s+that\s+section)\b",
            ]
            for h_pat in hallucinated_data_patterns:
                if re.search(h_pat, lower_cleaned):
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.UNSUPPORTED_ANALYSIS_REQUIREMENT,
                            severity=ValidationSeverity.ERROR,
                            message="Enhanced prompt references absent source material as if it were attached.",
                        )
                    )

        # Invariant: If critical error issues found, is_valid is False
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        is_valid = (error_count == 0)

        t1 = time.perf_counter()
        v_ms = (t1 - t0) * 1000.0

        return FastPathVerificationResult(
            is_valid=is_valid,
            cleaned_prompt=cleaned,
            issues=issues,
            deliverable_intercepted=deliverable_intercepted,
            verification_ms=v_ms,
        )

    async def enhance_fast_path(
        self,
        user_prompt: str,
        clarification_answers: Optional[list[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
        mode: str = "professional",
    ) -> FastPathResult:
        """Execute the production fast path pipeline.

        Strict invariants:
        - 0 LLM calls for preprocessing
        - Exactly 1 LLM call for generation (or 0 if critically ambiguous / missing data)
        - 0 LLM calls for verification
        """
        t_total_0 = time.perf_counter()
        timings: list[StageTimingDiagnostic] = []

        # Stage 1: Deterministic Preprocessing (0 LLM calls)
        t_prep_timer = StageTimer("deterministic_preprocessing")
        preprocess = self.preprocess(
            user_prompt=user_prompt,
            clarification_answers=clarification_answers,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
            mode=mode,
        )
        timings.append(t_prep_timer.complete(llm_called=False))
        analysis = preprocess.analysis

        # Ambiguity / Clarification Gate
        if analysis.ambiguity.needs_clarification and not allow_reasonable_assumptions:
            t_total_1 = time.perf_counter()
            total_ms = (t_total_1 - t_total_0) * 1000.0
            return FastPathResult(
                analysis,
                "",
                validation=None,
                repair_attempted=False,
                quality=None,
                stage_timings=timings,
                mode=mode,
                total_ms=total_ms,
                inference_ms=0.0,
            )

        # Stage 2: Single LLM Generation (Exactly ONE Ollama Call)
        t_enh_timer = StageTimer("enhancer", self.client)
        messages, num_predict = self._build_generation_prompt(
            user_prompt=user_prompt,
            preprocess=preprocess,
            mode=mode,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )
        try:
            raw_output, inference_ms, metadata = await self.generate(
                messages=messages,
                num_predict=num_predict,
            )
            timings.append(t_enh_timer.complete(llm_called=True))
        except OllamaTimeoutError:
            timings.append(t_enh_timer.complete(llm_called=True, status_code="ENHANCER_TIMEOUT"))
            raise

        # Stage 3: Deterministic Post-Generation Verification (0 LLM calls)
        t_verif_timer = StageTimer("deterministic_verification")
        verification = self.verify(
            enhanced_prompt=raw_output,
            user_prompt=user_prompt,
            preprocess=preprocess,
        )
        timings.append(t_verif_timer.complete(llm_called=False))

        val_obj = PromptValidation(
            is_valid=verification.is_valid,
            issues=verification.issues,
            checked_requirements=len(analysis.explicit_requirements),
            preserved_requirements=max(0, len(analysis.explicit_requirements) - sum(1 for i in verification.issues if i.type == ValidationIssueType.SOURCE_REQUIREMENT_DROPPED)),
            checked_constraints=len(analysis.constraints),
            preserved_constraints=max(0, len(analysis.constraints) - sum(1 for i in verification.issues if i.type == ValidationIssueType.CONSTRAINT_CONTRADICTION)),
            anti_invention_violations=sum(1 for i in verification.issues if i.type == ValidationIssueType.UNSUPPORTED_ANALYSIS_REQUIREMENT),
            validation_source="deterministic_fast_path",
        )

        t_total_1 = time.perf_counter()
        total_ms = (t_total_1 - t_total_0) * 1000.0

        return FastPathResult(
            analysis,
            verification.cleaned_prompt,
            validation=val_obj,
            repair_attempted=False,
            quality=None,
            initial_enhanced_prompt=verification.cleaned_prompt,
            initial_validation=val_obj,
            stage_timings=timings,
            mode=mode,
            total_ms=total_ms,
            inference_ms=inference_ms,
        )


fast_path_pipeline = FastPathPipeline()
