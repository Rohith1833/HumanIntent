"""
Comprehensive evaluation metrics calculation engine for Phase 10.
Computes deterministic accuracy, recall, hallucination, clarification,
inflation, latency, confusion matrices, and release gate statuses.
"""

import math
import re
from typing import Any, Dict, List, Tuple

from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkRunMetrics,
    CasePrediction,
    FailureType,
    ReleaseGateResult,
    ReleaseGateStatus,
)

# Forbidden meta / internal control expressions
CONTROL_LEAK_PATTERNS = [
    r"(?i)\bhumanintent\b",
    r"(?i)\bdo not assume these details without confirmation\b",
    r"(?i)\bask for clarification if needed\b",
    r"(?i)\bi have enhanced this prompt\b",
    r"(?i)\bi have removed the invented\b",
    r"</?(?:AUTHORITATIVE_USER_REQUEST|ORIGINAL_USER_REQUEST|STRUCTURED_INTENT_ANALYSIS|CATEGORY_TEMPLATE_GUIDANCE|ENHANCEMENT_DIRECTIVE|REPAIR_DIRECTIVE)>",
]


def _check_string_presence(target: str, item: str) -> bool:
    """Checks if substantial tokens of item are present in target text."""
    target_lower = target.lower()
    tokens = [w for w in re.findall(r"\b[a-z0-9_\-]+\b", item.lower()) if len(w) > 3 and w not in ["with", "from", "that", "this", "without", "into", "using"]]
    if not tokens:
        return item.lower() in target_lower
    return any(t in target_lower for t in tokens)


def compute_case_failures(
    case: BenchmarkCase,
    pred: CasePrediction,
) -> List[FailureType]:
    """Determines all failure taxonomy labels exhibited by a single prediction."""
    failures: List[FailureType] = []

    # Category Error
    if pred.detected_category and pred.detected_category.upper() != case.expected.category.upper():
        failures.append(FailureType.CATEGORY_ERROR)

    # Task Family Error
    if pred.detected_task_family and pred.detected_task_family.upper() != case.expected.task_family.upper():
        failures.append(FailureType.TASK_FAMILY_ERROR)

    # Clarification errors
    if case.expected.clarification_required and not pred.clarification_requested:
        failures.append(FailureType.UNDER_CLARIFICATION)
    elif not case.expected.clarification_required and pred.clarification_requested:
        failures.append(FailureType.OVER_CLARIFICATION)

    # Source dependency error
    if case.expected.source_required and not pred.clarification_requested:
        failures.append(FailureType.SOURCE_DEPENDENCY_ERROR)

    target_text = pred.enhanced_prompt

    # For non-clarification cases, evaluate target text
    if not pred.clarification_requested and target_text:
        # Requirement Recall
        for req in case.expected.must_preserve:
            if not _check_string_presence(target_text, req):
                failures.append(FailureType.MISSING_REQUIREMENT)
                break

        # Constraint Recall
        for c in case.expected.constraints:
            if not _check_string_presence(target_text, c):
                failures.append(FailureType.MISSING_CONSTRAINT)
                break

        # Quantity Preservation
        for q in case.expected.quantities:
            q_num = re.search(r"\b\d+\b", q)
            if q_num and q_num.group(0) not in target_text:
                failures.append(FailureType.QUANTITY_ERROR)
                break

        # Invented / Hallucinated Requirement
        for mni in case.expected.must_not_invent:
            if _check_string_presence(target_text, mni):
                failures.append(FailureType.INVENTED_REQUIREMENT)
                break

        # Meta Leakage
        for pat in CONTROL_LEAK_PATTERNS:
            if re.search(pat, target_text):
                failures.append(FailureType.META_LEAKAGE)
                break

        # Excessive Inflation on already good
        if case.expected.already_good and len(case.user_prompt) > 0:
            ratio = len(target_text) / len(case.user_prompt)
            if ratio > 2.5:
                failures.append(FailureType.EXCESSIVE_INFLATION)

    return list(set(failures))


def compute_confusion_matrix(
    cases: List[BenchmarkCase],
    predictions: List[CasePrediction],
) -> Dict[str, Dict[str, int]]:
    """Builds category confusion matrix (expected -> detected -> count)."""
    case_map = {c.case_id: c for c in cases}
    matrix: Dict[str, Dict[str, int]] = {}

    for p in predictions:
        case = case_map.get(p.case_id)
        if not case:
            continue
        exp_cat = case.expected.category.upper()
        det_cat = (p.detected_category or "UNKNOWN").upper()

        if exp_cat not in matrix:
            matrix[exp_cat] = {}
        matrix[exp_cat][det_cat] = matrix[exp_cat].get(det_cat, 0) + 1

    return matrix


def evaluate_release_gates(metrics: BenchmarkRunMetrics) -> List[ReleaseGateResult]:
    """Evaluates the 10 initial release gates against measured run metrics."""
    gates = [
        ReleaseGateResult(
            gate_name="Category Accuracy",
            target_threshold=">= 90.0%",
            actual_value=round(metrics.category_accuracy * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.category_accuracy >= 0.90 else ReleaseGateStatus.FAILED,
            description="Overall accuracy of intent category classification",
        ),
        ReleaseGateResult(
            gate_name="Task-Family Accuracy",
            target_threshold=">= 95.0%",
            actual_value=round(metrics.task_family_accuracy * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.task_family_accuracy >= 0.95 else ReleaseGateStatus.FAILED,
            description="Intent anchor action family consistency",
        ),
        ReleaseGateResult(
            gate_name="Explicit Requirement Recall",
            target_threshold=">= 95.0%",
            actual_value=round(metrics.requirement_recall * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.requirement_recall >= 0.95 else ReleaseGateStatus.FAILED,
            description="Preservation rate of explicit user requirements",
        ),
        ReleaseGateResult(
            gate_name="Constraint Recall",
            target_threshold=">= 95.0%",
            actual_value=round(metrics.constraint_recall * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.constraint_recall >= 0.95 else ReleaseGateStatus.FAILED,
            description="Preservation of negative and restrictive constraints",
        ),
        ReleaseGateResult(
            gate_name="Quantity Preservation",
            target_threshold=">= 97.0%",
            actual_value=round(metrics.quantity_preservation * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.quantity_preservation >= 0.97 else ReleaseGateStatus.FAILED,
            description="Preservation of explicit numerical limits and item counts",
        ),
        ReleaseGateResult(
            gate_name="Unsupported Requirement Rate",
            target_threshold="<= 3.0%",
            actual_value=round(metrics.unsupported_requirement_rate * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.unsupported_requirement_rate <= 0.03 else ReleaseGateStatus.FAILED,
            description="Rate of introducing unrequested features or hallucinations",
        ),
        ReleaseGateResult(
            gate_name="Objective Drift Rate",
            target_threshold="<= 2.0%",
            actual_value=round(metrics.objective_drift_rate * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.objective_drift_rate <= 0.02 else ReleaseGateStatus.FAILED,
            description="Rate of materially changing the requested user task",
        ),
        ReleaseGateResult(
            gate_name="Clarification F1",
            target_threshold=">= 90.0%",
            actual_value=round(metrics.clarification_f1 * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.clarification_f1 >= 0.90 else ReleaseGateStatus.FAILED,
            description="Harmonic mean of clarification precision and recall",
        ),
        ReleaseGateResult(
            gate_name="Over-Clarification Rate",
            target_threshold="<= 5.0%",
            actual_value=round(metrics.over_clarification_rate * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.over_clarification_rate <= 0.05 else ReleaseGateStatus.FAILED,
            description="Clear prompts incorrectly blocked for clarification",
        ),
        ReleaseGateResult(
            gate_name="Meta / Instruction Leakage",
            target_threshold="== 0.0%",
            actual_value=round(metrics.meta_leakage_rate * 100, 2),
            status=ReleaseGateStatus.PASSED if metrics.meta_leakage_rate == 0.0 else ReleaseGateStatus.FAILED,
            description="Presence of internal system control wording in enhanced output",
        ),
    ]
    return gates


def compute_run_metrics(
    cases: List[BenchmarkCase],
    predictions: List[CasePrediction],
    run_id: str = "run_001",
    model: str = "llama3:latest",
    benchmark_version: str = "humanintent-benchmark-v1.0",
) -> BenchmarkRunMetrics:
    """Aggregates all deterministic and semantic metrics across a benchmark run."""
    case_map = {c.case_id: c for c in cases}
    total = len(predictions)
    if total == 0:
        return BenchmarkRunMetrics(
            run_id=run_id,
            benchmark_version=benchmark_version,
            model=model,
            total_cases=0,
        )

    cat_correct = 0
    cat_evaluated = 0
    tf_correct = 0
    tf_evaluated = 0

    req_total = 0
    req_preserved = 0

    const_total = 0
    const_preserved = 0

    qty_total = 0
    qty_preserved = 0

    hallucinations = 0
    drift_count = 0
    meta_leaks = 0

    # Clarification counters
    tp_clar = 0
    fp_clar = 0
    fn_clar = 0
    tn_clar = 0

    # Source dependency
    src_total = 0
    src_correct = 0

    # Expansion ratios
    expansion_ratios: List[float] = []
    latencies: List[float] = []
    call_counts: List[int] = []

    # Already-good preservation
    already_good_total = 0
    already_good_preserved = 0

    failure_counts: Dict[str, int] = {}

    for p in predictions:
        case = case_map.get(p.case_id)
        if not case:
            continue

        latencies.append(p.latency_ms)
        call_counts.append(p.llm_call_count)

        # Categorization
        if p.detected_category:
            cat_evaluated += 1
            if p.detected_category.upper() == case.expected.category.upper():
                cat_correct += 1

        if p.detected_task_family:
            tf_evaluated += 1
            if p.detected_task_family.upper() == case.expected.task_family.upper():
                tf_correct += 1

        # Clarification
        if case.expected.clarification_required:
            if p.clarification_requested:
                tp_clar += 1
            else:
                fn_clar += 1
        else:
            if p.clarification_requested:
                fp_clar += 1
            else:
                tn_clar += 1

        # Source dependency
        if case.expected.source_required:
            src_total += 1
            if p.clarification_requested:
                src_correct += 1
        else:
            src_total += 1
            if not p.clarification_requested:
                src_correct += 1

        # Evaluate target text if not clarification
        if not p.clarification_requested and p.enhanced_prompt:
            text = p.enhanced_prompt
            orig = case.user_prompt

            if len(orig) > 0:
                ratio = len(text) / len(orig)
                expansion_ratios.append(round(ratio, 2))

            # Already good
            if case.expected.already_good:
                already_good_total += 1
                if len(orig) > 0 and (len(text) / len(orig)) <= 2.2:
                    already_good_preserved += 1

            # Requirements
            for req in case.expected.must_preserve:
                req_total += 1
                if _check_string_presence(text, req):
                    req_preserved += 1

            # Constraints
            for c in case.expected.constraints:
                const_total += 1
                if _check_string_presence(text, c):
                    const_preserved += 1

            # Quantities
            for q in case.expected.quantities:
                qty_total += 1
                q_num = re.search(r"\b\d+\b", q)
                if q_num and q_num.group(0) in text:
                    qty_preserved += 1
                elif not q_num and _check_string_presence(text, q):
                    qty_preserved += 1

            # Hallucinations / Unsupported requirements
            has_hallucination = False
            for mni in case.expected.must_not_invent:
                if _check_string_presence(text, mni):
                    has_hallucination = True
                    break
            if has_hallucination:
                hallucinations += 1

            # Meta leakage
            has_leak = any(bool(re.search(pat, text)) for pat in CONTROL_LEAK_PATTERNS)
            if has_leak:
                meta_leaks += 1

        # Record failure labels
        failures = compute_case_failures(case, p)
        for f in failures:
            failure_counts[f.value] = failure_counts.get(f.value, 0) + 1

    # Compute aggregate fractions
    category_acc = round(cat_correct / max(cat_evaluated, 1), 4) if cat_evaluated > 0 else 0.0
    task_family_acc = round(tf_correct / max(tf_evaluated, 1), 4) if tf_evaluated > 0 else 0.0

    req_recall = round(req_preserved / max(req_total, 1), 4) if req_total > 0 else 1.0
    const_recall = round(const_preserved / max(const_total, 1), 4) if const_total > 0 else 1.0
    qty_pres = round(qty_preserved / max(qty_total, 1), 4) if qty_total > 0 else 1.0

    unsupported_rate = round(hallucinations / max(total, 1), 4)
    drift_rate = round(drift_count / max(total, 1), 4)
    leakage_rate = round(meta_leaks / max(total, 1), 4)

    # Clarification metrics
    clar_prec = round(tp_clar / max(tp_clar + fp_clar, 1), 4) if (tp_clar + fp_clar) > 0 else 1.0
    clar_rec = round(tp_clar / max(tp_clar + fn_clar, 1), 4) if (tp_clar + fn_clar) > 0 else 1.0
    clar_f1 = (
        round(2 * (clar_prec * clar_rec) / max(clar_prec + clar_rec, 1e-9), 4)
        if (clar_prec + clar_rec) > 0
        else 0.0
    )
    over_clar_rate = round(fp_clar / max(fp_clar + tn_clar, 1), 4) if (fp_clar + tn_clar) > 0 else 0.0
    under_clar_rate = round(fn_clar / max(tp_clar + fn_clar, 1), 4) if (tp_clar + fn_clar) > 0 else 0.0
    src_acc = round(src_correct / max(src_total, 1), 4) if src_total > 0 else 1.0

    already_good_rate = (
        round(already_good_preserved / max(already_good_total, 1), 4)
        if already_good_total > 0
        else 1.0
    )

    # Percentiles for expansion, latencies, calls
    expansion_ratios.sort()
    med_exp = expansion_ratios[len(expansion_ratios) // 2] if expansion_ratios else 1.0
    p90_exp = expansion_ratios[min(int(math.ceil(0.90 * len(expansion_ratios))) - 1, len(expansion_ratios) - 1)] if expansion_ratios else 1.0
    p95_exp = expansion_ratios[min(int(math.ceil(0.95 * len(expansion_ratios))) - 1, len(expansion_ratios) - 1)] if expansion_ratios else 1.0

    latencies.sort()
    med_lat = latencies[len(latencies) // 2] if latencies else 0.0
    p90_lat = latencies[min(int(math.ceil(0.90 * len(latencies))) - 1, len(latencies) - 1)] if latencies else 0.0
    p95_lat = latencies[min(int(math.ceil(0.95 * len(latencies))) - 1, len(latencies) - 1)] if latencies else 0.0

    call_counts.sort()
    med_calls = float(call_counts[len(call_counts) // 2]) if call_counts else 1.0
    p90_calls = float(call_counts[min(int(math.ceil(0.90 * len(call_counts))) - 1, len(call_counts) - 1)]) if call_counts else 1.0
    p95_calls = float(call_counts[min(int(math.ceil(0.95 * len(call_counts))) - 1, len(call_counts) - 1)]) if call_counts else 1.0
    avg_calls = round(sum(call_counts) / max(len(call_counts), 1), 2) if call_counts else 1.0

    # Call budget breakdown (TASK 6)
    clear_calls = [p.total_llm_calls for p in predictions if not p.clarification_requested and p.enhanced_prompt]
    clar_calls = [p.total_llm_calls for p in predictions if p.clarification_requested]
    clear_avg = round(sum(clear_calls) / max(len(clear_calls), 1), 2) if clear_calls else 0.0
    clar_avg = round(sum(clar_calls) / max(len(clar_calls), 1), 2) if clar_calls else 0.0

    analysis_rep_count = sum(1 for p in predictions if p.analysis_repair_calls > 0)
    enh_rep_count = sum(1 for p in predictions if p.enhancement_repair_calls > 0)

    # Stage Latencies (TASK 7)
    def _pctiles(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"median": 0.0, "p90": 0.0, "p95": 0.0}
        s = sorted(arr)
        return {
            "median": round(s[len(s) // 2], 1),
            "p90": round(s[min(int(math.ceil(0.90 * len(s))) - 1, len(s) - 1)], 1),
            "p95": round(s[min(int(math.ceil(0.95 * len(s))) - 1, len(s) - 1)], 1),
        }

    stage_latencies = {
        "analysis": _pctiles([p.analysis_latency_ms for p in predictions if p.analysis_latency_ms > 0]),
        "enhancement": _pctiles([p.enhancement_latency_ms for p in predictions if p.enhancement_latency_ms > 0]),
        "validation": _pctiles([p.validation_latency_ms for p in predictions if p.validation_latency_ms > 0]),
        "scoring": _pctiles([p.scoring_latency_ms for p in predictions if p.scoring_latency_ms > 0]),
        "repair": _pctiles([p.repair_latency_ms for p in predictions if p.repair_latency_ms > 0]),
        "total": {"median": round(med_lat, 1), "p90": round(p90_lat, 1), "p95": round(p95_lat, 1)},
    }

    # Validation first pass success
    val_first_pass = round(
        sum(1 for p in predictions if p.validation_passed and p.enhancement_repair_calls == 0) / max(len(predictions), 1),
        4,
    )

    # Per-category breakdown (TASK 11)
    cat_metrics: Dict[str, Dict[str, float]] = {}
    cats_set = {c.expected.category.upper() for c in cases}
    for cat in sorted(cats_set):
        c_cases = [c for c in cases if c.expected.category.upper() == cat]
        c_preds = [p for p in predictions if case_map.get(p.case_id) and case_map[p.case_id].expected.category.upper() == cat]
        if not c_preds:
            continue
        c_correct = sum(1 for p in c_preds if p.detected_category and p.detected_category.upper() == cat)
        c_req_tot = sum(len(c.expected.must_preserve) for c in c_cases)
        c_req_pres = sum(
            sum(1 for req in c.expected.must_preserve if _check_string_presence(p.enhanced_prompt, req))
            for p, c in [(p, case_map[p.case_id]) for p in c_preds if not p.clarification_requested and p.enhanced_prompt]
        )
        c_const_tot = sum(len(c.expected.constraints) for c in c_cases)
        c_const_pres = sum(
            sum(1 for con in c.expected.constraints if _check_string_presence(p.enhanced_prompt, con))
            for p, c in [(p, case_map[p.case_id]) for p in c_preds if not p.clarification_requested and p.enhanced_prompt]
        )
        c_qty_tot = sum(len(c.expected.quantities) for c in c_cases)
        c_qty_pres = sum(
            sum(1 for q in c.expected.quantities if (re.search(r"\b\d+\b", q) and re.search(r"\b\d+\b", q).group(0) in p.enhanced_prompt) or _check_string_presence(p.enhanced_prompt, q))
            for p, c in [(p, case_map[p.case_id]) for p in c_preds if not p.clarification_requested and p.enhanced_prompt]
        )
        c_halluc = sum(
            1 for p, c in [(p, case_map[p.case_id]) for p in c_preds if not p.clarification_requested and p.enhanced_prompt]
            if any(_check_string_presence(p.enhanced_prompt, mni) for mni in c.expected.must_not_invent)
        )
        c_exp = [round(len(p.enhanced_prompt) / max(len(c.user_prompt), 1), 2) for p, c in [(p, case_map[p.case_id]) for p in c_preds if not p.clarification_requested and p.enhanced_prompt]]
        c_exp.sort()
        c_med_exp = c_exp[len(c_exp) // 2] if c_exp else 1.0

        cat_metrics[cat] = {
            "case_count": float(len(c_preds)),
            "category_accuracy": round(c_correct / max(len(c_preds), 1), 4),
            "requirement_recall": round(c_req_pres / max(c_req_tot, 1), 4) if c_req_tot > 0 else 1.0,
            "constraint_recall": round(c_const_pres / max(c_const_tot, 1), 4) if c_const_tot > 0 else 1.0,
            "quantity_preservation": round(c_qty_pres / max(c_qty_tot, 1), 4) if c_qty_tot > 0 else 1.0,
            "unsupported_requirement_rate": round(c_halluc / max(len(c_preds), 1), 4),
            "median_expansion": c_med_exp,
        }

    # Per-difficulty breakdown (TASK 12)
    diff_metrics: Dict[str, Dict[str, float]] = {}
    for diff in ["EASY", "MEDIUM", "HARD"]:
        d_cases = [c for c in cases if c.difficulty.value.upper() == diff]
        d_preds = [p for p in predictions if case_map.get(p.case_id) and case_map[p.case_id].difficulty.value.upper() == diff]
        if not d_preds:
            continue
        d_correct = sum(1 for p in d_preds if p.detected_category and p.detected_category.upper() == case_map[p.case_id].expected.category.upper())
        d_req_tot = sum(len(c.expected.must_preserve) for c in d_cases)
        d_req_pres = sum(
            sum(1 for req in c.expected.must_preserve if _check_string_presence(p.enhanced_prompt, req))
            for p, c in [(p, case_map[p.case_id]) for p in d_preds if not p.clarification_requested and p.enhanced_prompt]
        )
        d_const_tot = sum(len(c.expected.constraints) for c in d_cases)
        d_const_pres = sum(
            sum(1 for con in c.expected.constraints if _check_string_presence(p.enhanced_prompt, con))
            for p, c in [(p, case_map[p.case_id]) for p in d_preds if not p.clarification_requested and p.enhanced_prompt]
        )
        d_qty_tot = sum(len(c.expected.quantities) for c in d_cases)
        d_qty_pres = sum(
            sum(1 for q in c.expected.quantities if (re.search(r"\b\d+\b", q) and re.search(r"\b\d+\b", q).group(0) in p.enhanced_prompt) or _check_string_presence(p.enhanced_prompt, q))
            for p, c in [(p, case_map[p.case_id]) for p in d_preds if not p.clarification_requested and p.enhanced_prompt]
        )
        d_halluc = sum(
            1 for p, c in [(p, case_map[p.case_id]) for p in d_preds if not p.clarification_requested and p.enhanced_prompt]
            if any(_check_string_presence(p.enhanced_prompt, mni) for mni in c.expected.must_not_invent)
        )
        d_exp = [round(len(p.enhanced_prompt) / max(len(c.user_prompt), 1), 2) for p, c in [(p, case_map[p.case_id]) for p in d_preds if not p.clarification_requested and p.enhanced_prompt]]
        d_exp.sort()
        d_med_exp = d_exp[len(d_exp) // 2] if d_exp else 1.0

        diff_metrics[diff] = {
            "case_count": float(len(d_preds)),
            "category_accuracy": round(d_correct / max(len(d_preds), 1), 4),
            "requirement_recall": round(d_req_pres / max(d_req_tot, 1), 4) if d_req_tot > 0 else 1.0,
            "constraint_recall": round(d_const_pres / max(d_const_tot, 1), 4) if d_const_tot > 0 else 1.0,
            "quantity_preservation": round(d_qty_pres / max(d_qty_tot, 1), 4) if d_qty_tot > 0 else 1.0,
            "unsupported_requirement_rate": round(d_halluc / max(len(d_preds), 1), 4),
            "median_expansion": d_med_exp,
        }

    metrics = BenchmarkRunMetrics(
        run_id=run_id,
        benchmark_version=benchmark_version,
        model=model,
        total_cases=total,
        category_accuracy=category_acc,
        task_family_accuracy=task_family_acc,
        requirement_recall=req_recall,
        constraint_recall=const_recall,
        quantity_preservation=qty_pres,
        unsupported_requirement_rate=unsupported_rate,
        objective_drift_rate=drift_rate,
        clarification_precision=clar_prec,
        clarification_recall=clar_rec,
        clarification_f1=clar_f1,
        over_clarification_rate=over_clar_rate,
        under_clarification_rate=under_clar_rate,
        source_dependency_accuracy=src_acc,
        meta_leakage_rate=leakage_rate,
        already_good_preservation_rate=already_good_rate,
        median_expansion_ratio=med_exp,
        p90_expansion_ratio=p90_exp,
        p95_expansion_ratio=p95_exp,
        median_latency_ms=round(med_lat, 1),
        p90_latency_ms=round(p90_lat, 1),
        p95_latency_ms=round(p95_lat, 1),
        avg_llm_calls=avg_calls,
        median_calls=med_calls,
        p90_calls=p90_calls,
        p95_calls=p95_calls,
        clear_enhanced_avg_calls=clear_avg,
        needs_clarification_avg_calls=clar_avg,
        analysis_repair_cases=analysis_rep_count,
        enhancement_repair_cases=enh_rep_count,
        stage_latencies=stage_latencies,
        validation_first_pass_success=val_first_pass,
        failure_counts=failure_counts,
        category_metrics=cat_metrics,
        difficulty_metrics=diff_metrics,
    )

    # Attach evaluated release gates
    metrics.release_gates = evaluate_release_gates(metrics)
    return metrics
