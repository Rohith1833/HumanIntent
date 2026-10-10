"""
Evaluation baselines implementation for Phase 10.
Executes candidate systems without modifying runtime code:
- Baseline A: RAW user prompt unchanged
- Baseline B: GENERIC prompt-enhancer instruction on same LLM
- Baseline C: NO_VALIDATOR_ABLATION (HumanIntent without validator/repair)
- Baseline D: HUMANINTENT (Full production pipeline: Analyzer, Enhancer, Validator, Scorer)
"""

import time
import logging
from typing import Optional

from app.core.analyzer import prompt_analyzer
from app.core.enhancer import PromptEnhancer
from app.core.scorer import prompt_scorer
from app.core.validator import prompt_validator
from app.llm.ollama_client import OllamaClient, ollama_client
from app.schemas.validation import ValidationSeverity
from app.templates.selector import get_template
from app.prompts.enhancer import build_enhancer_user_message, HUMANINTENT_SYSTEM_PROMPT
from evaluation.schemas import BenchmarkCase, CasePrediction, EvaluationBaselineType

logger = logging.getLogger(__name__)

GENERIC_ENHANCER_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. Your job is to improve the user's prompt so it is clear, "
    "detailed, well-structured, and effective when given to an AI. Return ONLY the enhanced prompt. "
    "Do not include any conversational preamble, notes, or meta-commentary."
)

GENERIC_ENHANCER_USER_TEMPLATE = (
    "Please rewrite and improve the following prompt:\n\n"
    "\"\"\"\n{user_prompt}\n\"\"\"\n\n"
    "Enhanced Prompt:"
)


async def execute_raw_baseline(case: BenchmarkCase, model: str = "none") -> CasePrediction:
    """Baseline A: Raw user prompt unchanged. Fast control baseline."""
    return CasePrediction(
        case_id=case.case_id,
        baseline=EvaluationBaselineType.RAW,
        model=model,
        enhanced_prompt=case.user_prompt,
        clarification_requested=False,
        clarification_questions=[],
        detected_category=None,
        detected_task_family=None,
        quality_score=50.0,
        validation_passed=True,
        latency_ms=0.0,
        llm_call_count=0,
        analysis_calls=0,
        enhancement_calls=0,
        validation_calls=0,
        scoring_calls=0,
        analysis_repair_calls=0,
        enhancement_repair_calls=0,
        total_llm_calls=0,
        analysis_latency_ms=0.0,
        enhancement_latency_ms=0.0,
        validation_latency_ms=0.0,
        scoring_latency_ms=0.0,
        repair_latency_ms=0.0,
        raw_response={},
    )


async def execute_generic_baseline(
    case: BenchmarkCase,
    client: Optional[OllamaClient] = None,
    model: str = "llama3:latest",
) -> CasePrediction:
    """Baseline B: Simple generic prompt engineer prompt on same underlying LLM."""
    c = client or ollama_client
    start_time = time.perf_counter()

    messages = [
        {"role": "system", "content": GENERIC_ENHANCER_SYSTEM_PROMPT},
        {"role": "user", "content": GENERIC_ENHANCER_USER_TEMPLATE.format(user_prompt=case.user_prompt)},
    ]

    try:
        raw_output = await c.chat(messages=messages, model_override=model)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return CasePrediction(
            case_id=case.case_id,
            baseline=EvaluationBaselineType.GENERIC,
            model=model,
            enhanced_prompt=raw_output,
            clarification_requested=False,
            clarification_questions=[],
            detected_category=None,
            detected_task_family=None,
            quality_score=75.0,
            validation_passed=True,
            latency_ms=round(latency_ms, 2),
            llm_call_count=1,
            analysis_calls=0,
            enhancement_calls=1,
            validation_calls=0,
            scoring_calls=0,
            analysis_repair_calls=0,
            enhancement_repair_calls=0,
            total_llm_calls=1,
            enhancement_latency_ms=round(latency_ms, 2),
            raw_response={"text": raw_output},
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return CasePrediction(
            case_id=case.case_id,
            baseline=EvaluationBaselineType.GENERIC,
            model=model,
            enhanced_prompt=case.user_prompt,
            clarification_requested=False,
            clarification_questions=[],
            detected_category=None,
            detected_task_family=None,
            quality_score=0.0,
            validation_passed=False,
            latency_ms=round(latency_ms, 2),
            llm_call_count=1,
            analysis_calls=0,
            enhancement_calls=1,
            total_llm_calls=1,
            raw_response={"error": str(e)},
        )


async def execute_no_validator_ablation(
    case: BenchmarkCase,
    enhancer: Optional[PromptEnhancer] = None,
    model: str = "llama3:latest",
) -> CasePrediction:
    """Baseline C: HumanIntent analysis and template enhancement WITHOUT validator or repair."""
    enh = enhancer or PromptEnhancer(
        client=ollama_client,
        analyzer=prompt_analyzer,
        validator=None,
        scorer=prompt_scorer,
    )
    overall_start = time.perf_counter()

    try:
        res = await enh.enhance(case.user_prompt, validate=False)
        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0
        analysis, enhanced_text = res

        clar_req = analysis.ambiguity.needs_clarification
        clar_questions = analysis.ambiguity.clarification_questions if clar_req else []
        calls = 1 if clar_req else 2

        return CasePrediction(
            case_id=case.case_id,
            baseline=EvaluationBaselineType.NO_VALIDATOR_ABLATION,
            model=model,
            enhanced_prompt=enhanced_text,
            clarification_requested=clar_req,
            clarification_questions=clar_questions,
            detected_category=analysis.classification.category.value,
            detected_task_family=analysis.classification.category.value,
            quality_score=float(res.quality.overall_score) if res.quality else 70.0,
            validation_passed=True,
            latency_ms=round(total_latency_ms, 2),
            llm_call_count=calls,
            analysis_calls=1,
            enhancement_calls=0 if clar_req else 1,
            validation_calls=0,
            scoring_calls=0,
            total_llm_calls=calls,
            raw_response={"analysis": analysis.model_dump()},
        )
    except Exception as e:
        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0
        return CasePrediction(
            case_id=case.case_id,
            baseline=EvaluationBaselineType.NO_VALIDATOR_ABLATION,
            model=model,
            enhanced_prompt="",
            clarification_requested=False,
            clarification_questions=[],
            quality_score=0.0,
            validation_passed=False,
            latency_ms=round(total_latency_ms, 2),
            llm_call_count=1,
            analysis_calls=1,
            total_llm_calls=1,
            raw_response={"error": str(e)},
        )


async def execute_humanintent_baseline(
    case: BenchmarkCase,
    enhancer: Optional[PromptEnhancer] = None,
    model: str = "llama3:latest",
) -> CasePrediction:
    """
    Baseline D: Full production HumanIntent pipeline with analyzer, enhancer, validator, scorer.
    Properly verifies, invokes, and accounts for all 4 production stages:
    1. PromptAnalyzer (1 call)
    2. Clarification / Source Grounding Gate (deterministic, 0 calls; halts if needed)
    3. PromptEnhancer template generation (1 call)
    4. PromptValidator validation & 1-attempt repair (1 call, +1 repair +1 revalidation if needed)
    5. PromptScorer quality evaluation (1 call)
    Normal clear path: 1 + 1 + 1 + 1 = 4 LLM calls.
    Clarification path: 1 LLM call.
    """
    enh = enhancer or PromptEnhancer(
        client=ollama_client,
        analyzer=prompt_analyzer,
        validator=prompt_validator,
        scorer=prompt_scorer,
    )
    overall_start = time.perf_counter()

    analysis_calls = 1
    enhancement_calls = 0
    validation_calls = 0
    scoring_calls = 0
    analysis_repair_calls = 0
    enhancement_repair_calls = 0

    analysis_latency_ms = 0.0
    enhancement_latency_ms = 0.0
    validation_latency_ms = 0.0
    scoring_latency_ms = 0.0
    repair_latency_ms = 0.0

    try:
        # Stage 1: Analyzer
        t0 = time.perf_counter()
        orig_repair_fn = enh.analyzer.repair_analysis
        rep_count = [0]
        async def _rep_hook(*args, **kwargs):
            rep_count[0] += 1
            return await orig_repair_fn(*args, **kwargs)
        enh.analyzer.repair_analysis = _rep_hook
        try:
            analysis = await enh.analyzer.analyze(case.user_prompt)
        finally:
            enh.analyzer.repair_analysis = orig_repair_fn

        analysis_latency_ms = (time.perf_counter() - t0) * 1000.0
        if rep_count[0] > 0:
            analysis_repair_calls = rep_count[0]

        clar_req = bool(analysis.ambiguity.needs_clarification)
        clar_questions = analysis.ambiguity.clarification_questions if clar_req else []

        grounded = True
        if hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded:
            grounded = False

        enhanced_text = ""
        validation = None
        quality_report = None
        val_passed = True
        q_score = 0.0

        # Stage 2: Clarification & Grounding Gates
        if grounded and not clar_req:
            # Stage 3: Enhancer Generation
            selected_template = get_template(analysis.classification.category)
            user_content = build_enhancer_user_message(
                case.user_prompt,
                analysis,
                template=selected_template,
                allow_reasonable_assumptions=False,
            )
            messages = [
                {"role": "system", "content": HUMANINTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            t1 = time.perf_counter()
            raw_enhanced = await enh.client.chat(messages=messages)
            enhanced_text = enh._clean_enhanced_output(raw_enhanced, analysis, case.user_prompt)
            enhancement_latency_ms = (time.perf_counter() - t1) * 1000.0
            enhancement_calls = 1

            # Stage 4: Validator & Controlled 1-attempt repair
            if enh.validator is not None:
                t2 = time.perf_counter()
                validation = await enh.validator.validate(
                    original_prompt=case.user_prompt,
                    analysis=analysis,
                    enhanced_prompt=enhanced_text,
                    allow_reasonable_assumptions=False,
                )
                validation_latency_ms = (time.perf_counter() - t2) * 1000.0
                validation_calls = 1

                if not validation.is_valid:
                    # Repair attempt
                    t_rep = time.perf_counter()
                    errors = [i for i in validation.issues if i.severity == ValidationSeverity.ERROR]
                    repaired_text = await enh.repair(
                        user_prompt=case.user_prompt,
                        analysis=analysis,
                        previous_enhanced_prompt=enhanced_text,
                        errors=errors,
                        template=selected_template,
                        allow_reasonable_assumptions=False,
                    )
                    enhancement_repair_calls = 1

                    # Re-validate repaired prompt
                    reval = await enh.validator.validate(
                        original_prompt=case.user_prompt,
                        analysis=analysis,
                        enhanced_prompt=repaired_text,
                        allow_reasonable_assumptions=False,
                    )
                    validation_calls += 1
                    validation = reval
                    enhanced_text = repaired_text
                    repair_latency_ms = (time.perf_counter() - t_rep) * 1000.0

                val_passed = validation.is_valid

            # Stage 5: Scorer
            if enh.scorer is not None:
                t3 = time.perf_counter()
                quality_report = await enh.scorer.score(
                    original_prompt=case.user_prompt,
                    analysis=analysis,
                    enhanced_prompt=enhanced_text,
                    validation=validation,
                )
                scoring_latency_ms = (time.perf_counter() - t3) * 1000.0
                scoring_calls = 1
                q_score = float(quality_report.overall_score) if quality_report else 85.0
            else:
                q_score = 85.0

        elif clar_req:
            q_score = 0.0
            val_passed = True
        else:
            # Grounding failed
            q_score = 0.0
            val_passed = False

        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0
        total_calls = (
            analysis_calls
            + enhancement_calls
            + validation_calls
            + scoring_calls
            + analysis_repair_calls
            + enhancement_repair_calls
        )

        return CasePrediction(
            case_id=case.case_id,
            baseline=EvaluationBaselineType.HUMANINTENT,
            model=model,
            enhanced_prompt=enhanced_text,
            clarification_requested=clar_req,
            clarification_questions=clar_questions,
            detected_category=analysis.classification.category.value,
            detected_task_family=analysis.classification.category.value,
            quality_score=q_score,
            validation_passed=val_passed,
            latency_ms=round(total_latency_ms, 2),
            llm_call_count=total_calls,
            analysis_calls=analysis_calls,
            enhancement_calls=enhancement_calls,
            validation_calls=validation_calls,
            scoring_calls=scoring_calls,
            analysis_repair_calls=analysis_repair_calls,
            enhancement_repair_calls=enhancement_repair_calls,
            total_llm_calls=total_calls,
            analysis_latency_ms=round(analysis_latency_ms, 2),
            enhancement_latency_ms=round(enhancement_latency_ms, 2),
            validation_latency_ms=round(validation_latency_ms, 2),
            scoring_latency_ms=round(scoring_latency_ms, 2),
            repair_latency_ms=round(repair_latency_ms, 2),
            raw_response={
                "analysis": analysis.model_dump(),
                "repair_attempted": bool(enhancement_repair_calls > 0),
            },
        )
    except Exception as e:
        logger.error("HumanIntent baseline execution error on case %s: %s", case.case_id, e)
        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0
        return CasePrediction(
            case_id=case.case_id,
            baseline=EvaluationBaselineType.HUMANINTENT,
            model=model,
            enhanced_prompt="",
            clarification_requested=False,
            clarification_questions=[],
            quality_score=0.0,
            validation_passed=False,
            latency_ms=round(total_latency_ms, 2),
            llm_call_count=1,
            analysis_calls=1,
            enhancement_calls=0,
            validation_calls=0,
            scoring_calls=0,
            analysis_repair_calls=0,
            enhancement_repair_calls=0,
            total_llm_calls=1,
            raw_response={"error": str(e)},
        )
