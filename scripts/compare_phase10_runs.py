#!/usr/bin/env python3
"""
Phase 10 vs Phase 10.1 Post-Hardening Comparison & Reporting Tool.
Programmatically compares original Phase 10 baseline against Phase 10.1 post-hardening results.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(repo_root, "backend")
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.schemas import BenchmarkCase, CasePrediction, BenchmarkRunMetrics
from evaluation.metrics import compute_run_metrics, compute_confusion_matrix, compute_case_failures, evaluate_release_gates
from evaluation.bootstrap import compute_benchmark_bootstrap_ci


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: str, baseline_filter: str = "humanintent") -> List[CasePrediction]:
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = CasePrediction.model_validate_json(line)
            if p.baseline.value == baseline_filter:
                preds.append(p)
    return preds


def load_cases(path: str = "benchmarks/v1/benchmark.jsonl") -> List[BenchmarkCase]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(BenchmarkCase.model_validate_json(line))
    return cases


def format_delta(old: float, new: float, higher_is_better: bool = True, is_percentage: bool = True) -> tuple[str, str, str]:
    abs_delta = new - old
    if abs(old) > 1e-9:
        rel_change = (abs_delta / abs(old)) * 100.0
        rel_str = f"{rel_change:+.2f}%"
    else:
        rel_str = "N/A"

    if is_percentage:
        old_str = f"{old * 100:.2f}%"
        new_str = f"{new * 100:.2f}%"
        delta_str = f"{abs_delta * 100:+.2f} pp"
    else:
        old_str = f"{old:.2f}"
        new_str = f"{new:.2f}"
        delta_str = f"{abs_delta:+.2f}"

    if abs(abs_delta) < 1e-5:
        verdict = "UNCHANGED"
    elif (abs_delta > 0 and higher_is_better) or (abs_delta < 0 and not higher_is_better):
        verdict = "BETTER"
    else:
        verdict = "WORSE"

    return old_str, new_str, delta_str, rel_str, verdict


def generate_full_comparison_report(
    old_dir: str = "evaluation_results/full_v1",
    new_dir: str = "evaluation_results/post_hardening_v1",
    cases_path: str = "benchmarks/v1/benchmark.jsonl",
) -> Dict[str, Any]:
    old_metrics_file = os.path.join(old_dir, "metrics.json")
    new_metrics_file = os.path.join(new_dir, "metrics.json")
    new_ckpt_file = os.path.join(new_dir, "predictions_checkpoint.jsonl")

    cases = load_cases(cases_path)
    old_m = load_json(old_metrics_file)

    if os.path.exists(new_metrics_file):
        new_m = load_json(new_metrics_file)
        new_preds = load_predictions(os.path.join(new_dir, "predictions.jsonl"))
        new_m_obj = BenchmarkRunMetrics.model_validate(new_m)
    elif os.path.exists(new_ckpt_file):
        new_preds = load_predictions(new_ckpt_file)
        new_m_obj = compute_run_metrics(cases, new_preds, run_id="eval_post_hardening", model="llama3:latest")
        new_m = new_m_obj.model_dump()
    else:
        raise FileNotFoundError("No post-hardening results found!")

    old_preds = load_predictions(os.path.join(old_dir, "predictions.jsonl"))

    # Comparison metrics definitions: (metric_key, higher_is_better, is_percentage, display_name)
    metrics_to_compare = [
        ("category_accuracy", True, True, "Category Accuracy"),
        ("task_family_accuracy", True, True, "Task-Family Accuracy"),
        ("requirement_recall", True, True, "Explicit Requirement Recall"),
        ("constraint_recall", True, True, "Constraint Recall"),
        ("quantity_preservation", True, True, "Quantity Preservation"),
        ("unsupported_requirement_rate", False, True, "Unsupported Requirement Rate"),
        ("objective_drift_rate", False, True, "Objective Drift Rate"),
        ("clarification_precision", True, True, "Clarification Precision"),
        ("clarification_recall", True, True, "Clarification Recall"),
        ("clarification_f1", True, True, "Clarification F1"),
        ("over_clarification_rate", False, True, "Over-Clarification Rate"),
        ("under_clarification_rate", False, True, "Under-Clarification Rate"),
        ("source_dependency_accuracy", True, True, "Source-Dependency Accuracy"),
        ("meta_leakage_rate", False, True, "Meta / Instruction Leakage Rate"),
        ("validation_first_pass_success", True, True, "Validation First-Pass Success"),
        ("median_expansion_ratio", False, False, "Median Expansion Ratio"),
    ]

    comp_rows = []
    for key, higher_is_better, is_percentage, display_name in metrics_to_compare:
        old_val = float(old_m.get(key, 0.0))
        new_val = float(new_m.get(key, 0.0))
        o_str, n_str, d_str, r_str, verdict = format_delta(old_val, new_val, higher_is_better, is_percentage)
        comp_rows.append({
            "metric": display_name,
            "old": o_str,
            "new": n_str,
            "delta": d_str,
            "relative": r_str,
            "verdict": verdict,
        })

    # Confusion matrix
    conf_mat = compute_confusion_matrix(cases, new_preds)

    # Categories breakdown
    old_cat = old_m.get("category_metrics", {})
    new_cat = new_m.get("category_metrics", {})
    all_cats = sorted(list(set(list(old_cat.keys()) + list(new_cat.keys()))))
    cat_rows = []
    for c in all_cats:
        oc = old_cat.get(c, {})
        nc = new_cat.get(c, {})
        cat_rows.append({
            "category": c,
            "cases": int(nc.get("case_count", 0)),
            "old_acc": f"{oc.get('category_accuracy', 0.0)*100:.1f}%",
            "new_acc": f"{nc.get('category_accuracy', 0.0)*100:.1f}%",
            "old_req": f"{oc.get('requirement_recall', 0.0)*100:.1f}%",
            "new_req": f"{nc.get('requirement_recall', 0.0)*100:.1f}%",
            "old_const": f"{oc.get('constraint_recall', 0.0)*100:.1f}%",
            "new_const": f"{nc.get('constraint_recall', 0.0)*100:.1f}%",
            "old_unsupp": f"{oc.get('unsupported_requirement_rate', 0.0)*100:.1f}%",
            "new_unsupp": f"{nc.get('unsupported_requirement_rate', 0.0)*100:.1f}%",
            "old_qty": f"{oc.get('quantity_preservation', 0.0)*100:.1f}%",
            "new_qty": f"{nc.get('quantity_preservation', 0.0)*100:.1f}%",
        })

    # Difficulty breakdown
    old_diff = old_m.get("difficulty_metrics", {})
    new_diff = new_m.get("difficulty_metrics", {})
    diff_rows = []
    for d in ["EASY", "MEDIUM", "HARD"]:
        od = old_diff.get(d, {})
        nd = new_diff.get(d, {})
        diff_rows.append({
            "difficulty": d,
            "cases": int(nd.get("case_count", 0)),
            "old_acc": f"{od.get('category_accuracy', 0.0)*100:.1f}%",
            "new_acc": f"{nd.get('category_accuracy', 0.0)*100:.1f}%",
            "old_req": f"{od.get('requirement_recall', 0.0)*100:.1f}%",
            "new_req": f"{nd.get('requirement_recall', 0.0)*100:.1f}%",
            "old_const": f"{od.get('constraint_recall', 0.0)*100:.1f}%",
            "new_const": f"{nd.get('constraint_recall', 0.0)*100:.1f}%",
            "old_unsupp": f"{od.get('unsupported_requirement_rate', 0.0)*100:.1f}%",
            "new_unsupp": f"{nd.get('unsupported_requirement_rate', 0.0)*100:.1f}%",
        })

    # Release gates
    gates_data = evaluate_release_gates(new_m_obj)

    # Failure taxonomy
    old_failures = old_m.get("failure_counts", {})
    new_failures = new_m.get("failure_counts", {})
    all_fail_types = sorted(list(set(list(old_failures.keys()) + list(new_failures.keys()))), key=lambda x: new_failures.get(x, 0), reverse=True)
    fail_rows = []
    for ft in all_fail_types:
        of = old_failures.get(ft, 0)
        nf = new_failures.get(ft, 0)
        fail_rows.append({
            "failure_type": ft,
            "old_count": of,
            "new_count": nf,
            "delta": nf - of,
        })

    # Performance & Timing Analysis
    latencies = [p.latency_ms / 1000.0 for p in new_preds if p.latency_ms > 0]
    analyzer_lats = [p.analysis_latency_ms / 1000.0 for p in new_preds]
    enhancer_lats = [p.enhancement_latency_ms / 1000.0 for p in new_preds]
    validator_lats = [p.validation_latency_ms / 1000.0 for p in new_preds]
    scorer_lats = [p.scoring_latency_ms / 1000.0 for p in new_preds]
    calls = [p.total_llm_calls for p in new_preds]

    mean_tot = float(np.mean(latencies)) if latencies else 0.0
    med_tot = float(np.median(latencies)) if latencies else 0.0
    p90_tot = float(np.percentile(latencies, 90)) if latencies else 0.0
    p95_tot = float(np.percentile(latencies, 95)) if latencies else 0.0

    sum_tot = sum(latencies) if sum(latencies) > 0 else 1.0
    scorer_pct = (sum(scorer_lats) / sum_tot) * 100.0
    analyzer_pct = (sum(analyzer_lats) / sum_tot) * 100.0
    enhancer_pct = (sum(enhancer_lats) / sum_tot) * 100.0
    validator_pct = (sum(validator_lats) / sum_tot) * 100.0

    repair_cases = sum(1 for p in new_preds if p.enhancement_repair_calls > 0 or p.analysis_repair_calls > 0)
    clar_halt_cases = sum(1 for p in new_preds if p.clarification_requested)
    timeout_cases = sum(1 for p in new_preds if p.quality_score == 0.0 and not p.clarification_requested and not p.validation_passed and "error" in (p.raw_response or {}))

    return {
        "comparison_table": comp_rows,
        "confusion_matrix": conf_mat,
        "category_breakdown": cat_rows,
        "difficulty_breakdown": diff_rows,
        "release_gates": [g.model_dump() for g in gates_data],
        "failure_taxonomy": fail_rows,
        "performance": {
            "cases_completed": len(new_preds),
            "mean_total_latency_sec": round(mean_tot, 2),
            "median_total_latency_sec": round(med_tot, 2),
            "p90_total_latency_sec": round(p90_tot, 2),
            "p95_total_latency_sec": round(p95_tot, 2),
            "mean_llm_calls": round(float(np.mean(calls)), 2) if calls else 0.0,
            "median_llm_calls": int(np.median(calls)) if calls else 0,
            "p90_llm_calls": int(np.percentile(calls, 90)) if calls else 0,
            "p95_llm_calls": int(np.percentile(calls, 95)) if calls else 0,
            "repair_rate_pct": round((repair_cases / max(len(new_preds), 1)) * 100.0, 2),
            "clarification_halt_pct": round((clar_halt_cases / max(len(new_preds), 1)) * 100.0, 2),
            "timeout_rate_pct": round((timeout_cases / max(len(new_preds), 1)) * 100.0, 2),
            "scorer_pct_total": round(scorer_pct, 2),
            "analyzer_pct_total": round(analyzer_pct, 2),
            "enhancer_pct_total": round(enhancer_pct, 2),
            "validator_pct_total": round(validator_pct, 2),
        },
        "old_summary": old_m,
        "new_summary": new_m,
    }


if __name__ == "__main__":
    report = generate_full_comparison_report()
    print(json.dumps(report, indent=2))
