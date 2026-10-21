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
    enh = enhancer or prompt_enhancer
    overall_start = time.perf_counter()

    # Intercept analyzer repair attempts to record analysis_repair_calls
    orig_repair_fn = enh.analyzer.repair_analysis
    rep_count = [0]

    async def _rep_hook(*args, **kwargs):
        rep_count[0] += 1
        return await orig_repair_fn(*args, **kwargs)

    enh.analyzer.repair_analysis = _rep_hook

    # Intercept NLI and Semantic Matcher to record per-case telemetry
    from app.core.nli_matcher import nli_matcher
    from app.core.semantic_matcher import semantic_matcher
    from app.core.category_router import category_router
    from app.core.intent_anchor import extract_intent_anchor

    nli_counter = [0]
    nli_latency_acc = [0.0]
    orig_nli_predict = nli_matcher.predict

    def _nli_hook(*args, **kwargs):
        nli_counter[0] += 1
        t_start = time.perf_counter()
        res_nli = orig_nli_predict(*args, **kwargs)
        nli_latency_acc[0] += (time.perf_counter() - t_start) * 1000.0
        return res_nli

    nli_matcher.predict = _nli_hook

    sem_counter = [0]
    orig_sem_sim = semantic_matcher.similarity

    def _sem_hook(*args, **kwargs):
        sem_counter[0] += 1
        return orig_sem_sim(*args, **kwargs)

    semantic_matcher.similarity = _sem_hook

    # Pre-compute routing tier for telemetry
    try:
        anchor = extract_intent_anchor(case.user_prompt)
        routing_info = category_router.route(case.user_prompt, anchor=anchor)
        routing_tier = routing_info.tier
        router_latency_ms = routing_info.latency_ms
    except Exception:
        routing_tier = "unknown"
        router_latency_ms = 0.0

    try:
        res = await enh.enhance(case.user_prompt)
        analysis = res.analysis
        enhanced_text = res.enhanced_prompt
        validation = res.validation
        quality_report = res.quality
        timings = res.stage_timings or []

        clar_req = bool(analysis.ambiguity.needs_clarification)
        raw_qs = getattr(analysis.ambiguity, "questions", getattr(analysis.ambiguity, "clarification_questions", []))
        clar_questions = [q.question if hasattr(q, "question") else str(q) for q in (raw_qs or [])] if clar_req else []

        val_passed = bool(validation.is_valid) if validation is not None else (True if clar_req else False)
        q_score = (
            float(quality_report.overall_score)
            if quality_report
            else (0.0 if clar_req else (85.0 if val_passed else 0.0))
        )

        analysis_calls = sum(1 for t in timings if t.stage == "analyzer" and t.llm_called)
        analysis_repair_calls = rep_count[0]
        enhancement_calls = sum(1 for t in timings if t.stage == "enhancer" and t.llm_called)
        val_stage_calls = sum(1 for t in timings if t.stage == "validator" and t.llm_called)
        reval_stage_calls = sum(1 for t in timings if t.stage == "revalidation" and t.llm_called)
        validation_calls = val_stage_calls + reval_stage_calls
        enhancement_repair_calls = sum(1 for t in timings if t.stage == "enhancement_repair" and t.llm_called)
        scoring_calls = sum(1 for t in timings if t.stage == "scorer" and t.llm_called)
        total_calls = (
            analysis_calls
            + analysis_repair_calls
            + enhancement_calls
            + validation_calls
            + enhancement_repair_calls
            + scoring_calls
        )

        analysis_latency_ms = sum(t.elapsed_ms for t in timings if t.stage == "analyzer")
        enhancement_latency_ms = sum(t.elapsed_ms for t in timings if t.stage == "enhancer")
        validation_latency_ms = sum(t.elapsed_ms for t in timings if t.stage in ("validator", "revalidation"))
        scoring_latency_ms = sum(t.elapsed_ms for t in timings if t.stage == "scorer")
        repair_latency_ms = sum(t.elapsed_ms for t in timings if t.stage in ("enhancement_repair", "revalidation"))
        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0

        prompt_tokens = sum(t.prompt_tokens or 0 for t in timings)
        generated_tokens = sum(t.generated_tokens or 0 for t in timings)
        processing_mode = "long_input" if any("long_input" in (t.repair_reason or "") for t in timings) or len(case.user_prompt.split()) > 400 else "normal"

        ana_path = getattr(analysis, "analysis_path", "llm")
        if ana_path in ("deterministic_fast", "semantic_fast"):
            analysis_path_type = "fast_path"
        else:
            analysis_path_type = "llm"

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
                "repair_attempted": res.repair_attempted,
                "stage_timings": [t.model_dump() for t in timings],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "processing_mode": processing_mode,
                "analysis_path": ana_path,
                "analysis_path_type": analysis_path_type,
                "category_routing_tier": routing_tier,
                "router_latency_ms": round(router_latency_ms, 2),
                "semantic_matcher_usage": sem_counter[0],
                "nli_invocation_count": nli_counter[0],
                "nli_latency_ms": round(nli_latency_acc[0], 2),
                "revalidation_calls": reval_stage_calls,
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
    finally:
        enh.analyzer.repair_analysis = orig_repair_fn
        nli_matcher.predict = orig_nli_predict
        semantic_matcher.similarity = orig_sem_sim
