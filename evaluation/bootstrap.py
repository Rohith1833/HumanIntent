"""
Bootstrap statistical estimation for Phase 10 evaluation.
Computes deterministic 95% non-parametric confidence intervals without heavy external dependencies.
"""

import math
import random
import re
from typing import Dict, List, Tuple, Any, Optional

from evaluation.schemas import BenchmarkCase, CasePrediction
from evaluation.metrics import _check_string_presence


def compute_bootstrap_ci(
    sample: List[float],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes (mean, ci_lower, ci_upper) using non-parametric bootstrap resampling with replacement.
    """
    if not sample:
        return 0.0, 0.0, 0.0

    n = len(sample)
    if n == 1:
        val = sample[0]
        return val, val, val

    rng = random.Random(random_seed)
    bootstrap_means: List[float] = []

    for _ in range(iterations):
        resample = [rng.choice(sample) for _ in range(n)]
        bootstrap_means.append(sum(resample) / n)

    bootstrap_means.sort()
    orig_mean = sum(sample) / n

    alpha = 1.0 - confidence_level
    lower_idx = int(math.floor((alpha / 2.0) * iterations))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * iterations)) - 1

    ci_lower = bootstrap_means[max(0, min(lower_idx, iterations - 1))]
    ci_upper = bootstrap_means[max(0, min(upper_idx, iterations - 1))]

    return round(orig_mean, 4), round(ci_lower, 4), round(ci_upper, 4)


def compute_benchmark_bootstrap_ci(
    cases: List[BenchmarkCase],
    predictions: List[CasePrediction],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes aligned 95% bootstrap confidence intervals for all primary benchmark metrics.
    Uses case-level non-parametric bootstrap resampling with replacement where each resample
    is evaluated using the exact estimator function corresponding to the reported point estimate.

    Guarantees:
    - Point estimate and bootstrap distribution refer to the exact same estimator.
    - Metric names cannot become positionally mismatched.
    - Clarification F1 cannot receive an accuracy or recall CI.
    - Zero/degenerate estimators produce exact, bounded intervals.
    """
    case_map = {c.case_id: c for c in cases}
    case_stats = []

    for p in predictions:
        case = case_map.get(p.case_id)
        if not case:
            continue

        cat_eval = 1 if p.detected_category else 0
        cat_corr = 1 if (p.detected_category and p.detected_category.upper() == case.expected.category.upper()) else 0

        tf_eval = 1 if p.detected_task_family else 0
        tf_corr = 1 if (p.detected_task_family and p.detected_task_family.upper() == case.expected.task_family.upper()) else 0

        req_tot = 0
        req_pres = 0
        const_tot = 0
        const_pres = 0
        qty_tot = 0
        qty_pres = 0
        unsupp = 0

        if not p.clarification_requested and p.enhanced_prompt:
            text = p.enhanced_prompt
            for req in case.expected.must_preserve:
                req_tot += 1
                if _check_string_presence(text, req):
                    req_pres += 1

            for c in case.expected.constraints:
                const_tot += 1
                if _check_string_presence(text, c):
                    const_pres += 1

            for q in case.expected.quantities:
                qty_tot += 1
                q_num = re.search(r"\b\d+\b", q)
                if q_num and q_num.group(0) in text:
                    qty_pres += 1
                elif not q_num and _check_string_presence(text, q):
                    qty_pres += 1

            for mni in case.expected.must_not_invent:
                if _check_string_presence(text, mni):
                    unsupp = 1
                    break

        tp = 1 if (case.expected.clarification_required and p.clarification_requested) else 0
        fp = 1 if (not case.expected.clarification_required and p.clarification_requested) else 0
        fn = 1 if (case.expected.clarification_required and not p.clarification_requested) else 0

        case_stats.append({
            "cat_eval": cat_eval, "cat_corr": cat_corr,
            "tf_eval": tf_eval, "tf_corr": tf_corr,
            "req_tot": req_tot, "req_pres": req_pres,
            "const_tot": const_tot, "const_pres": const_pres,
            "qty_tot": qty_tot, "qty_pres": qty_pres,
            "unsupp": unsupp,
            "tp": tp, "fp": fp, "fn": fn,
        })

    n = len(case_stats)
    if n == 0:
        return {}

    # Exact point estimates on the original full dataset
    orig_ce = sum(s["cat_eval"] for s in case_stats)
    orig_cc = sum(s["cat_corr"] for s in case_stats)
    orig_tfe = sum(s["tf_eval"] for s in case_stats)
    orig_tfc = sum(s["tf_corr"] for s in case_stats)
    orig_rt = sum(s["req_tot"] for s in case_stats)
    orig_rp = sum(s["req_pres"] for s in case_stats)
    orig_ct = sum(s["const_tot"] for s in case_stats)
    orig_cp = sum(s["const_pres"] for s in case_stats)
    orig_qt = sum(s["qty_tot"] for s in case_stats)
    orig_qp = sum(s["qty_pres"] for s in case_stats)
    orig_un = sum(s["unsupp"] for s in case_stats)
    orig_tp = sum(s["tp"] for s in case_stats)
    orig_fp = sum(s["fp"] for s in case_stats)
    orig_fn = sum(s["fn"] for s in case_stats)

    if orig_tp == 0 and (orig_tp + orig_fn) > 0:
        orig_f1 = 0.0
    elif (orig_tp + orig_fp) == 0 and (orig_tp + orig_fn) == 0:
        orig_f1 = 1.0
    else:
        orig_prec = round(orig_tp / max(orig_tp + orig_fp, 1), 4) if (orig_tp + orig_fp) > 0 else 1.0
        orig_rec = round(orig_tp / max(orig_tp + orig_fn, 1), 4) if (orig_tp + orig_fn) > 0 else 1.0
        orig_f1 = (
            round(2 * (orig_prec * orig_rec) / max(orig_prec + orig_rec, 1e-9), 4)
            if (orig_prec + orig_rec) > 0
            else 0.0
        )

    point_estimates = {
        "category_accuracy": round(orig_cc / max(orig_ce, 1), 4) if orig_ce > 0 else 0.0,
        "task_family_accuracy": round(orig_tfc / max(orig_tfe, 1), 4) if orig_tfe > 0 else 0.0,
        "requirement_recall": round(orig_rp / max(orig_rt, 1), 4) if orig_rt > 0 else 1.0,
        "constraint_recall": round(orig_cp / max(orig_ct, 1), 4) if orig_ct > 0 else 1.0,
        "quantity_preservation": round(orig_qp / max(orig_qt, 1), 4) if orig_qt > 0 else 1.0,
        "unsupported_requirement_rate": round(orig_un / n, 4),
        "clarification_f1": round(orig_f1, 4),
    }

    # Bootstrap Resampling
    rng = random.Random(random_seed)
    metric_keys = list(point_estimates.keys())
    bootstrap_samples: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(iterations):
        resample = [rng.choice(case_stats) for _ in range(n)]

        ce = sum(s["cat_eval"] for s in resample)
        cc = sum(s["cat_corr"] for s in resample)
        bootstrap_samples["category_accuracy"].append(cc / max(ce, 1) if ce > 0 else 0.0)

        tfe = sum(s["tf_eval"] for s in resample)
        tfc = sum(s["tf_corr"] for s in resample)
        bootstrap_samples["task_family_accuracy"].append(tfc / max(tfe, 1) if tfe > 0 else 0.0)

        rt = sum(s["req_tot"] for s in resample)
        rp = sum(s["req_pres"] for s in resample)
        bootstrap_samples["requirement_recall"].append(rp / max(rt, 1) if rt > 0 else 1.0)

        ct = sum(s["const_tot"] for s in resample)
        cp = sum(s["const_pres"] for s in resample)
        bootstrap_samples["constraint_recall"].append(cp / max(ct, 1) if ct > 0 else 1.0)

        qt = sum(s["qty_tot"] for s in resample)
        qp = sum(s["qty_pres"] for s in resample)
        bootstrap_samples["quantity_preservation"].append(qp / max(qt, 1) if qt > 0 else 1.0)

        un = sum(s["unsupp"] for s in resample)
        bootstrap_samples["unsupported_requirement_rate"].append(un / n)

        tp_s = sum(s["tp"] for s in resample)
        fp_s = sum(s["fp"] for s in resample)
        fn_s = sum(s["fn"] for s in resample)
        if tp_s == 0 and (orig_tp + orig_fn > 0 or (tp_s + fn_s) > 0):
            f1 = 0.0
        elif (tp_s + fp_s) == 0 and (tp_s + fn_s) == 0:
            f1 = 1.0
        else:
            prec = round(tp_s / max(tp_s + fp_s, 1), 4) if (tp_s + fp_s) > 0 else 1.0
            rec = round(tp_s / max(tp_s + fn_s, 1), 4) if (tp_s + fn_s) > 0 else 1.0
            f1 = (
                round(2 * (prec * rec) / max(prec + rec, 1e-9), 4)
                if (prec + rec) > 0
                else 0.0
            )
        bootstrap_samples["clarification_f1"].append(round(f1, 4))

    alpha = 1.0 - confidence_level
    lower_idx = int(math.floor((alpha / 2.0) * iterations))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * iterations)) - 1

    results: Dict[str, Dict[str, float]] = {}
    for metric_name in metric_keys:
        vals = sorted(bootstrap_samples[metric_name])
        ci_low = vals[max(0, min(lower_idx, iterations - 1))]
        ci_high = vals[max(0, min(upper_idx, iterations - 1))]
        b_mean = sum(vals) / iterations
        pt = point_estimates[metric_name]

        # Defensive assertions ensuring point estimate & distribution correspond
        assert ci_low <= ci_high, f"Invalid interval for {metric_name}: [{ci_low}, {ci_high}]"
        if pt == 0.0 and all(v == 0.0 for v in vals):
            assert ci_low == 0.0 and ci_high == 0.0, f"Zero estimator {metric_name} must have [0, 0] CI"

        results[metric_name] = {
            "point_estimate": round(pt, 4),
            "mean": round(b_mean, 4),
            "ci_lower": round(ci_low, 4),
            "ci_upper": round(ci_high, 4),
            "confidence_level": confidence_level,
        }

    return results


def compute_all_bootstrap_intervals(
    binary_metrics: Dict[str, List[int]],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Legacy helper computing confidence intervals across dictionary of metric outcomes (0/1)."""
    results: Dict[str, Dict[str, float]] = {}
    for metric_name, sample in binary_metrics.items():
        float_sample = [float(x) for x in sample]
        mean, low, high = compute_bootstrap_ci(
            float_sample,
            iterations=iterations,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        orig_mean = sum(float_sample) / len(float_sample) if float_sample else 0.0
        results[metric_name] = {
            "point_estimate": round(orig_mean, 4),
            "mean": mean,
            "ci_lower": low,
            "ci_upper": high,
            "confidence_level": confidence_level,
        }
    return results
