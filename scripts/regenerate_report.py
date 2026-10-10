#!/usr/bin/env python3
"""
Regenerates Phase 10 evaluation artifacts (metrics.json, report.md) directly from
existing predictions without re-running any LLM inference.
"""

import os
import sys
import json
import logging

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(repo_root, "backend")
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.benchmark_loader import load_benchmark
from evaluation.schemas import CasePrediction
from evaluation.metrics import compute_run_metrics, compute_confusion_matrix, evaluate_release_gates
from evaluation.bootstrap import compute_benchmark_bootstrap_ci
from evaluation.report import EvaluationReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("regenerate_report")


def main():
    benchmark_path = os.path.join("benchmarks", "v1", "benchmark.jsonl")
    metadata_path = os.path.join("benchmarks", "v1", "metadata.json")
    results_dir = os.path.join("evaluation_results", "full_v1")
    predictions_path = os.path.join(results_dir, "predictions.jsonl")

    logger.info("Loading frozen benchmark: %s", benchmark_path)
    cases = load_benchmark(benchmark_path=benchmark_path, metadata_path=metadata_path)
    logger.info("Loaded %d validated benchmark cases.", len(cases))

    logger.info("Loading existing predictions from: %s", predictions_path)
    with open(predictions_path, "r", encoding="utf-8") as f:
        all_preds = [CasePrediction.model_validate_json(line) for line in f if line.strip()]

    by_baseline = {}
    for p in all_preds:
        by_baseline.setdefault(p.baseline.value, []).append(p)

    logger.info("Loaded baseline counts: %s", {k: len(v) for k, v in by_baseline.items()})

    # Compute baseline metrics
    baseline_metrics = {}
    for b_name, preds in by_baseline.items():
        baseline_metrics[b_name] = compute_run_metrics(
            cases=cases,
            predictions=preds,
            run_id=f"eval_{b_name}",
            model="llama3:latest",
        )

    hi_metrics = baseline_metrics["humanintent"]

    # Compute aligned bootstrap confidence intervals
    logger.info("Computing aligned 95% bootstrap confidence intervals (1000 resamples)...")
    ci_results = compute_benchmark_bootstrap_ci(
        cases=cases,
        predictions=by_baseline["humanintent"],
        iterations=1000,
        confidence_level=0.95,
        random_seed=42,
    )
    hi_metrics.confidence_intervals = ci_results

    # Preserve pairwise and position bias from existing metrics.json if present
    existing_metrics_path = os.path.join(results_dir, "metrics.json")
    pairwise_summary = {
        "vs_raw_win": 0.0, "vs_raw_loss": 0.0, "vs_raw_tie": 1.0, "vs_raw_denominator": "0 / 22",
        "vs_gen_win": 0.0, "vs_gen_loss": 0.0, "vs_gen_tie": 1.0, "vs_gen_denominator": "0 / 22",
        "position_bias_tested": 22, "position_bias_consistent": 22, "position_bias_consistency_rate": 1.0,
    }
    if os.path.exists(existing_metrics_path):
        with open(existing_metrics_path, "r", encoding="utf-8") as f:
            old_m = json.load(f)
            hi_metrics.vs_raw_wins = old_m.get("vs_raw_wins", 0)
            hi_metrics.vs_raw_total = old_m.get("vs_raw_total", 22)
            hi_metrics.win_rate_vs_raw = old_m.get("win_rate_vs_raw", 0.0)
            hi_metrics.vs_generic_wins = old_m.get("vs_generic_wins", 0)
            hi_metrics.vs_generic_total = old_m.get("vs_generic_total", 22)
            hi_metrics.win_rate_vs_generic = old_m.get("win_rate_vs_generic", 0.0)
            hi_metrics.position_bias_cases_tested = old_m.get("position_bias_cases_tested", 22)
            hi_metrics.position_bias_consistent_cases = old_m.get("position_bias_consistent_cases", 22)
            hi_metrics.position_bias_consistency_rate = old_m.get("position_bias_consistency_rate", 1.0)
            if "stage_latencies" in old_m:
                hi_metrics.stage_latencies = old_m["stage_latencies"]
            if "avg_llm_calls" in old_m:
                hi_metrics.avg_llm_calls = old_m["avg_llm_calls"]
                hi_metrics.median_calls = old_m["median_calls"]
                hi_metrics.p90_calls = old_m["p90_calls"]
                hi_metrics.p95_calls = old_m["p95_calls"]
                hi_metrics.clear_enhanced_avg_calls = old_m["clear_enhanced_avg_calls"]
                hi_metrics.needs_clarification_avg_calls = old_m["needs_clarification_avg_calls"]
                hi_metrics.analysis_repair_cases = old_m["analysis_repair_cases"]
                hi_metrics.enhancement_repair_cases = old_m["enhancement_repair_cases"]

    # Confusion matrix
    conf_mat = compute_confusion_matrix(cases, by_baseline["humanintent"])

    # Re-export
    reporter = EvaluationReportGenerator(results_dir)
    m_path = reporter.export_metrics_json(hi_metrics)
    r_path = reporter.export_markdown_report(
        metrics=hi_metrics,
        baseline_metrics=baseline_metrics,
        pairwise_summary=pairwise_summary,
        confusion_matrix=conf_mat,
        total_benchmark_cases=len(cases),
    )
    g_path = reporter.export_release_gates_json(hi_metrics.release_gates)

    logger.info("Successfully updated metrics at: %s", m_path)
    logger.info("Successfully updated report at: %s", r_path)
    logger.info("Successfully updated release gates at: %s", g_path)

    print("\n" + "=" * 70)
    print("CORRECTED BOOTSTRAP 95% CONFIDENCE INTERVALS TABLE")
    print("=" * 70)
    print(f"{'Metric':30} | {'Point Est':10} | {'Bootstrap Mean':14} | {'95% CI':20}")
    print("-" * 70)
    for k, v in ci_results.items():
        print(f"{k:30} | {v['point_estimate']:>9.2%} | {v['mean']:>13.2%} | [{v['ci_lower']:>6.2%}, {v['ci_upper']:>6.2%}]")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
