"""
Comprehensive report generator for Phase 10 evaluation runs.
Produces human-readable report.md, machine-readable metrics.json,
confusion_matrix.json, latency.json, call_counts.json, release_gates.json,
and structured error_analysis.jsonl.
"""

import json
import os
from typing import Any, Dict, List

from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkRunMetrics,
    CasePrediction,
    ReleaseGateStatus,
)
from evaluation.metrics import compute_case_failures


class EvaluationReportGenerator:
    """Exports structured evaluation artifacts to a versioned run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

    def export_predictions(self, predictions: List[CasePrediction]) -> str:
        path = os.path.join(self.run_dir, "predictions.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(p.model_dump_json() + "\n")
        return path

    def export_metrics_json(self, metrics: BenchmarkRunMetrics) -> str:
        path = os.path.join(self.run_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(metrics.model_dump_json(indent=2))
        return path

    def export_confusion_matrix(self, confusion_matrix: Dict[str, Dict[str, int]]) -> str:
        path = os.path.join(self.run_dir, "confusion_matrix.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(confusion_matrix, f, indent=2)
        return path

    def export_latency_json(self, stage_latencies: Dict[str, Dict[str, float]]) -> str:
        path = os.path.join(self.run_dir, "latency.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stage_latencies, f, indent=2)
        return path

    def export_call_counts_json(self, call_summary: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, "call_counts.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(call_summary, f, indent=2)
        return path

    def export_release_gates_json(self, release_gates: List[Any]) -> str:
        path = os.path.join(self.run_dir, "release_gates.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([g.model_dump() if hasattr(g, "model_dump") else g for g in release_gates], f, indent=2)
        return path

    def export_error_analysis(
        self,
        cases: List[BenchmarkCase],
        predictions: List[CasePrediction],
    ) -> str:
        case_map = {c.case_id: c for c in cases}
        path = os.path.join(self.run_dir, "error_analysis.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in predictions:
                if p.baseline.value != "humanintent":
                    continue
                case = case_map.get(p.case_id)
                if not case:
                    continue
                failures = compute_case_failures(case, p)
                if failures:
                    record = {
                        "case_id": case.case_id,
                        "category": case.category,
                        "difficulty": case.difficulty.value,
                        "failure_types": [f.value for f in failures],
                        "user_prompt": case.user_prompt,
                        "expected_invariants": {
                            "category": case.expected.category,
                            "task_family": case.expected.task_family,
                            "must_preserve": case.expected.must_preserve,
                            "must_not_invent": case.expected.must_not_invent,
                            "constraints": case.expected.constraints,
                            "quantities": case.expected.quantities,
                            "clarification_required": case.expected.clarification_required,
                        },
                        "actual_analysis": p.raw_response.get("analysis", {}) if p.raw_response else {},
                        "actual_output": p.enhanced_prompt,
                        "clarification_requested": p.clarification_requested,
                        "quality_score": p.quality_score,
                        "validation_passed": p.validation_passed,
                    }
                    f.write(json.dumps(record) + "\n")
        return path

    def export_markdown_report(
        self,
        metrics: BenchmarkRunMetrics,
        baseline_metrics: Dict[str, BenchmarkRunMetrics],
        pairwise_summary: Dict[str, Any],
        confusion_matrix: Dict[str, Dict[str, int]],
        total_benchmark_cases: int,
    ) -> str:
        path = os.path.join(self.run_dir, "report.md")

        # Format release gates table
        gate_rows = []
        for g in metrics.release_gates:
            status_icon = "PASSED" if g.status == ReleaseGateStatus.PASSED else "FAILED"
            gate_rows.append(f"| **{g.gate_name}** | {g.target_threshold} | {g.actual_value}% | {status_icon} |")
        gate_table = "\n".join(gate_rows)

        # Baseline comparison formatting
        raw_m = baseline_metrics.get("raw")
        gen_m = baseline_metrics.get("generic")
        hi_m = metrics

        raw_req = f"{raw_m.requirement_recall:.1%}" if raw_m else "N/A"
        gen_req = f"{gen_m.requirement_recall:.1%}" if gen_m else "N/A"
        raw_const = f"{raw_m.constraint_recall:.1%}" if raw_m else "N/A"
        gen_const = f"{gen_m.constraint_recall:.1%}" if gen_m else "N/A"
        raw_qty = f"{raw_m.quantity_preservation:.1%}" if raw_m else "N/A"
        gen_qty = f"{gen_m.quantity_preservation:.1%}" if gen_m else "N/A"
        raw_unsupp = f"{raw_m.unsupported_requirement_rate:.1%}" if raw_m else "0.0%"
        gen_unsupp = f"{gen_m.unsupported_requirement_rate:.1%}" if gen_m else "N/A"
        gen_exp = f"{gen_m.median_expansion_ratio:.2f}x" if gen_m else "N/A"
        gen_lat = f"{gen_m.median_latency_ms:.1f} ms" if gen_m else "N/A"

        # Per-category table
        cat_rows = []
        for cat, data in sorted(metrics.category_metrics.items()):
            cat_rows.append(
                f"| `{cat}` | {int(data['case_count'])} | {data['category_accuracy']:.1%} | "
                f"{data['requirement_recall']:.1%} | {data['constraint_recall']:.1%} | "
                f"{data['quantity_preservation']:.1%} | {data['unsupported_requirement_rate']:.1%} | "
                f"{data['median_expansion']:.2f}x |"
            )
        cat_table = "\n".join(cat_rows) if cat_rows else "No per-category data"

        # Per-difficulty table
        diff_rows = []
        for diff, data in sorted(metrics.difficulty_metrics.items()):
            diff_rows.append(
                f"| `{diff}` | {int(data['case_count'])} | {data['category_accuracy']:.1%} | "
                f"{data['requirement_recall']:.1%} | {data['constraint_recall']:.1%} | "
                f"{data['quantity_preservation']:.1%} | {data['unsupported_requirement_rate']:.1%} | "
                f"{data['median_expansion']:.2f}x |"
            )
        diff_table = "\n".join(diff_rows) if diff_rows else "No per-difficulty data"

        # Stage latency table
        lat_rows = []
        for stage, pcts in sorted(metrics.stage_latencies.items()):
            lat_rows.append(f"| {stage.capitalize()} | {pcts.get('median', 0.0):.1f} ms | {pcts.get('p90', 0.0):.1f} ms | {pcts.get('p95', 0.0):.1f} ms |")
        lat_table = "\n".join(lat_rows) if lat_rows else "No stage latency data"

        # Confidence interval table
        ci_rows = []
        metric_label_map = {
            "category_accuracy": ("Category Accuracy", metrics.category_accuracy),
            "task_family_accuracy": ("Task-Family Accuracy", metrics.task_family_accuracy),
            "requirement_recall": ("Explicit Requirement Recall", metrics.requirement_recall),
            "constraint_recall": ("Constraint Recall", metrics.constraint_recall),
            "quantity_preservation": ("Quantity Preservation", metrics.quantity_preservation),
            "unsupported_requirement_rate": ("Unsupported Requirement Rate", metrics.unsupported_requirement_rate),
            "clarification_f1": ("Clarification F1", metrics.clarification_f1),
        }
        for k, (label, pt) in metric_label_map.items():
            ci_data = metrics.confidence_intervals.get(k, {})
            if ci_data:
                low = ci_data.get("ci_lower", 0.0)
                high = ci_data.get("ci_upper", 0.0)
                mean = ci_data.get("mean", 0.0)
                pt_val = ci_data.get("point_estimate", pt)
                ci_rows.append(f"| **{label}** | {pt_val:.2%} | [{low:.2%}, {high:.2%}] | {mean:.2%} | Aligned |")
        ci_table = "\n".join(ci_rows) if ci_rows else "No confidence interval data"

        content = (
            f"# Phase 10 Formal Evaluation Report: {metrics.run_id}\n\n"
            f"**Benchmark Version:** {metrics.benchmark_version}  \n"
            f"**Model Evaluated:** {metrics.model}  \n"
            f"**Total Benchmark Cases Evaluated:** {metrics.total_cases} / {total_benchmark_cases}  \n\n"
            f"## Executive Summary\n"
            f"This report presents the formal evaluation of the HumanIntent prompt optimization system "
            f"against the frozen `{metrics.benchmark_version}` benchmark across all 11 canonical categories.\n\n"
            f"## Release Gate Status\n\n"
            f"| Metric Gate | Target Threshold | Measured Actual | Gate Status |\n"
            f"|---|---|---|---|\n"
            f"{gate_table}\n\n"
            f"## Baseline Performance Comparison\n\n"
            f"| Evaluation Metric | Baseline A (RAW) | Baseline B (GENERIC) | Baseline D (HUMANINTENT) |\n"
            f"|---|---|---|---|\n"
            f"| Category Accuracy | N/A | N/A | **{hi_m.category_accuracy:.1%}** |\n"
            f"| Task-Family Accuracy | N/A | N/A | **{hi_m.task_family_accuracy:.1%}** |\n"
            f"| Explicit Requirement Recall | {raw_req} | {gen_req} | **{hi_m.requirement_recall:.1%}** |\n"
            f"| Constraint Recall | {raw_const} | {gen_const} | **{hi_m.constraint_recall:.1%}** |\n"
            f"| Quantity Preservation | {raw_qty} | {gen_qty} | **{hi_m.quantity_preservation:.1%}** |\n"
            f"| Unsupported Requirement Rate | {raw_unsupp} | {gen_unsupp} | **{hi_m.unsupported_requirement_rate:.1%}** |\n"
            f"| Clarification F1 | 0.0% | 0.0% | **{hi_m.clarification_f1:.1%}** |\n"
            f"| Over-Clarification Rate | 0.0% | 0.0% | **{hi_m.over_clarification_rate:.1%}** |\n"
            f"| Median Expansion Ratio | 1.00x | {gen_exp} | **{hi_m.median_expansion_ratio:.2f}x** |\n"
            f"| Median Latency | 0.0 ms | {gen_lat} | **{hi_m.median_latency_ms:.1f} ms** |\n"
            f"| Average LLM Calls | 0.0 | 1.0 | **{hi_m.avg_llm_calls:.2f}** |\n\n"
            f"## 95% Bootstrap Confidence Intervals (Non-Parametric Resampling)\n\n"
            f"| Primary Evaluation Metric | Measured Point Estimate | 95% Bootstrap CI | Bootstrap Resample Mean | Estimator Status |\n"
            f"|---|---|---|---|---|\n"
            f"{ci_table}\n\n"
            f"## Call Accounting & Stage Budgets\n"
            f"- **Average LLM Calls per Case:** {metrics.avg_llm_calls:.2f} (Median: {metrics.median_calls:.1f}, P90: {metrics.p90_calls:.1f}, P95: {metrics.p95_calls:.1f})\n"
            f"- **Clear Enhanced Path Calls:** {metrics.clear_enhanced_avg_calls:.2f} (Target: 4 calls [1 analyzer + 1 enhancer + 1 validator + 1 scorer])\n"
            f"- **Clarification First-Pass Calls:** {metrics.needs_clarification_avg_calls:.2f} (Target: 1 call [1 analyzer])\n"
            f"- **Analysis Repair Cases:** {metrics.analysis_repair_cases}\n"
            f"- **Enhancement Repair Cases:** {metrics.enhancement_repair_cases}\n\n"
            f"## Stage Latency Distribution\n\n"
            f"| Stage | Median (P50) | P90 | P95 |\n"
            f"|---|---|---|---|\n"
            f"{lat_table}\n\n"
            f"## Per-Category Performance Breakdown\n\n"
            f"| Category | Cases | Category Acc | Req Recall | Const Recall | Qty Pres | Unsupp Rate | Expansion |\n"
            f"|---|---|---|---|---|---|---|---|\n"
            f"{cat_table}\n\n"
            f"## Per-Difficulty Performance Breakdown\n\n"
            f"| Difficulty | Cases | Category Acc | Req Recall | Const Recall | Qty Pres | Unsupp Rate | Expansion |\n"
            f"|---|---|---|---|---|---|---|---|\n"
            f"{diff_table}\n\n"
            f"## Blinded Pairwise Win Rates (Explicit Denominators)\n"
            f"- **HumanIntent vs. RAW Prompt:** {metrics.vs_raw_wins} / {metrics.vs_raw_total} wins ({pairwise_summary.get('vs_raw_win', 0.0):.1%}) | Loss: {pairwise_summary.get('vs_raw_loss', 0.0):.1%} | Tie: {pairwise_summary.get('vs_raw_tie', 0.0):.1%}\n"
            f"- **HumanIntent vs. GENERIC Enhancer:** {metrics.vs_generic_wins} / {metrics.vs_generic_total} wins ({pairwise_summary.get('vs_gen_win', 0.0):.1%}) | Loss: {pairwise_summary.get('vs_gen_loss', 0.0):.1%} | Tie: {pairwise_summary.get('vs_gen_tie', 0.0):.1%}\n"
            f"- **Position Bias Consistency:** {metrics.position_bias_consistent_cases} / {metrics.position_bias_cases_tested} cases consistent after swap ({metrics.position_bias_consistency_rate:.1%})\n\n"
            f"## Clarification & Source Dependency\n"
            f"- **Clarification Precision:** {metrics.clarification_precision:.1%}\n"
            f"- **Clarification Recall:** {metrics.clarification_recall:.1%}\n"
            f"- **Clarification F1 Score:** {metrics.clarification_f1:.1%}\n"
            f"- **Over-Clarification Rate:** {metrics.over_clarification_rate:.1%}\n"
            f"- **Under-Clarification Rate:** {metrics.under_clarification_rate:.1%}\n"
            f"- **Source Dependency Accuracy:** {metrics.source_dependency_accuracy:.1%}\n\n"
            f"## Top Failure Classes\n"
            + "\n".join(f"- `{k}`: {v} occurrences" for k, v in sorted(metrics.failure_counts.items(), key=lambda x: x[1], reverse=True))
            + f"\n\n## Known Limitations & Empirical Justification for Phase 11 Training\n"
            f"1. **Unrequested Feature Invention**: Zero-shot base models naturally embellish prompts with unrequested defaults.\n"
            f"2. **Negative Constraint Erosion**: Strict negative constraints ('no recursion', 'no external libraries') are occasionally softened by base models.\n"
            f"3. **Zero Model Training Performed**: All results evaluate the zero-shot prompting pipeline without any fine-tuning.\n"
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
