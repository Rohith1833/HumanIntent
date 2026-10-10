"""
Core orchestrator for Phase 10 evaluation runs.
Supports multi-baseline execution, resumable checkpoints, dry-run mode,
blinded pairwise judging, position-bias measurement, and metric reporting.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.core.enhancer import PromptEnhancer, prompt_enhancer
from app.llm.ollama_client import ollama_client
from evaluation.baselines import (
    execute_generic_baseline,
    execute_humanintent_baseline,
    execute_no_validator_ablation,
    execute_raw_baseline,
)
from evaluation.benchmark_loader import load_benchmark
from evaluation.bootstrap import compute_benchmark_bootstrap_ci
from evaluation.config import EvaluationConfig
from evaluation.judges import evaluate_pairwise_blinded
from evaluation.metrics import compute_confusion_matrix, compute_run_metrics
from evaluation.report import EvaluationReportGenerator
from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkRunMetrics,
    CasePrediction,
    EvaluationBaselineType,
)

logger = logging.getLogger("evaluation.runner")


class EvaluationRunner:
    """End-to-end evaluation runner."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        # Use full production prompt_enhancer with analyzer, validator, and scorer
        self.enhancer = prompt_enhancer
        self.reporter = EvaluationReportGenerator(config.output_dir)

    def _get_checkpoint_path(self) -> str:
        return os.path.join(self.config.output_dir, "predictions_checkpoint.jsonl")

    async def run(self) -> Dict[str, Any]:
        logger.info("Initializing Phase 10 evaluation...")

        # 1. Load and cryptographically verify frozen benchmark
        cases = load_benchmark(
            benchmark_path=self.config.benchmark_path,
            metadata_path=self.config.metadata_path,
            categories=self.config.categories,
            difficulty=self.config.difficulty,
            limit=self.config.limit,
            sample_per_category=self.config.sample_per_category,
        )

        # 2. Dry Run Mode
        if self.config.dry_run:
            logger.info("[DRY RUN] Benchmark verified successfully. Total eligible cases: %d", len(cases))
            print("\n=======================================================")
            print("PHASE 10 EVALUATION DRY RUN COMPLETE")
            print("=======================================================")
            print(f"Benchmark Hash Verified: True")
            print(f"Eligible Cases:          {len(cases)}")
            print(f"Active Baselines:        {', '.join(self.config.baselines)}")
            print(f"Model:                   {self.config.model}")
            print("Zero Ollama LLM calls made.")
            print("=======================================================\n")
            return {
                "status": "dry_run_passed",
                "eligible_cases": len(cases),
                "total_cases": len(cases),
                "benchmark_verified": True,
                "baselines": self.config.baselines,
                "model": self.config.model,
                "dry_run": True,
            }

        # 3. Check for resumable predictions
        checkpoint_path = self._get_checkpoint_path()
        completed_keys = set()
        all_predictions: List[CasePrediction] = []

        if os.path.exists(checkpoint_path) and not self.config.force:
            logger.info("Resuming evaluation from checkpoint: %s", checkpoint_path)
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = CasePrediction.model_validate_json(line)
                        all_predictions.append(p)
                        completed_keys.add((p.case_id, p.baseline.value))
            logger.info("Loaded %d existing predictions from checkpoint.", len(all_predictions))

        # 4. Execute predictions for active baselines
        with open(checkpoint_path, "a", encoding="utf-8") as ckpt_file:
            for idx, case in enumerate(cases, 1):
                logger.info("Evaluating case %d/%d: %s (%s - %s)", idx, len(cases), case.case_id, case.category, case.difficulty.value)

                for b_name in self.config.baselines:
                    key = (case.case_id, b_name)
                    if key in completed_keys:
                        continue

                    pred: Optional[CasePrediction] = None
                    if b_name == "raw":
                        pred = await execute_raw_baseline(case, model=self.config.model)
                    elif b_name == "generic":
                        pred = await execute_generic_baseline(case, client=ollama_client, model=self.config.model)
                    elif b_name == "no_validator_ablation":
                        pred = await execute_no_validator_ablation(case, enhancer=self.enhancer, model=self.config.model)
                    elif b_name == "humanintent":
                        pred = await execute_humanintent_baseline(case, enhancer=self.enhancer, model=self.config.model)

                    if pred:
                        all_predictions.append(pred)
                        completed_keys.add(key)
                        ckpt_file.write(pred.model_dump_json() + "\n")
                        ckpt_file.flush()

        # 5. Compute metrics by baseline
        by_baseline: Dict[str, List[CasePrediction]] = {}
        for p in all_predictions:
            by_baseline.setdefault(p.baseline.value, []).append(p)

        baseline_metrics: Dict[str, BenchmarkRunMetrics] = {}
        for b_name, preds in by_baseline.items():
            run_id = f"eval_{b_name}"
            m = compute_run_metrics(
                cases=cases,
                predictions=preds,
                run_id=run_id,
                model=self.config.model,
            )
            baseline_metrics[b_name] = m

        hi_metrics = baseline_metrics.get("humanintent", BenchmarkRunMetrics(run_id="eval_humanintent", benchmark_version="v1.0", model=self.config.model, total_cases=0))

        # 6. Blinded Pairwise Comparisons & Position-Bias Measurement (TASK 16, 17, 19)
        hi_preds = {p.case_id: p for p in by_baseline.get("humanintent", [])}
        raw_preds = {p.case_id: p for p in by_baseline.get("raw", [])}
        gen_preds = {p.case_id: p for p in by_baseline.get("generic", [])}

        vs_raw_wins = 0
        vs_raw_losses = 0
        vs_raw_ties = 0

        vs_gen_wins = 0
        vs_gen_losses = 0
        vs_gen_ties = 0

        position_bias_tested = 0
        position_bias_consistent = 0

        # Stratified pairwise judging subset across all 11 categories (2 per category = 22 cases)
        cats_count = {}
        pairwise_sample_cases = []
        for c in cases:
            c_cat = c.category.value if hasattr(c.category, "value") else str(c.category)
            if cats_count.get(c_cat, 0) < 2:
                pairwise_sample_cases.append(c)
                cats_count[c_cat] = cats_count.get(c_cat, 0) + 1
        if not pairwise_sample_cases:
            pairwise_sample_cases = cases[:22]

        for c in pairwise_sample_cases:
            h_p = hi_preds.get(c.case_id)
            r_p = raw_preds.get(c.case_id)
            g_p = gen_preds.get(c.case_id)

            if h_p and r_p:
                pair_res = await evaluate_pairwise_blinded(
                    case=c,
                    candidate_a=h_p,
                    candidate_b=r_p,
                    client=ollama_client,
                    judge_model=self.config.judge_model,
                )
                if pair_res.decoded_winner == "humanintent":
                    vs_raw_wins += 1
                elif pair_res.decoded_winner == "raw":
                    vs_raw_losses += 1
                else:
                    vs_raw_ties += 1

            if h_p and g_p:
                # Forward comparison (A=first, B=second)
                pair_fwd = await evaluate_pairwise_blinded(
                    case=c,
                    candidate_a=h_p,
                    candidate_b=g_p,
                    client=ollama_client,
                    judge_model=self.config.judge_model,
                    swap_override=False,
                )
                # Reversed comparison (B=first, A=second) to test position bias
                pair_rev = await evaluate_pairwise_blinded(
                    case=c,
                    candidate_a=h_p,
                    candidate_b=g_p,
                    client=ollama_client,
                    judge_model=self.config.judge_model,
                    swap_override=True,
                )

                position_bias_tested += 1
                if pair_fwd.decoded_winner == pair_rev.decoded_winner:
                    position_bias_consistent += 1

                if pair_fwd.decoded_winner == "humanintent":
                    vs_gen_wins += 1
                elif pair_fwd.decoded_winner == "generic":
                    vs_gen_losses += 1
                else:
                    vs_gen_ties += 1

        n_pairs = max(len(pairwise_sample_cases), 1)
        pairwise_summary = {
            "vs_raw_win": round(vs_raw_wins / n_pairs, 4),
            "vs_raw_loss": round(vs_raw_losses / n_pairs, 4),
            "vs_raw_tie": round(vs_raw_ties / n_pairs, 4),
            "vs_raw_denominator": f"{vs_raw_wins} / {n_pairs}",
            "vs_gen_win": round(vs_gen_wins / n_pairs, 4),
            "vs_gen_loss": round(vs_gen_losses / n_pairs, 4),
            "vs_gen_tie": round(vs_gen_ties / n_pairs, 4),
            "vs_gen_denominator": f"{vs_gen_wins} / {n_pairs}",
            "position_bias_tested": position_bias_tested,
            "position_bias_consistent": position_bias_consistent,
            "position_bias_consistency_rate": round(position_bias_consistent / max(position_bias_tested, 1), 4),
        }

        hi_metrics.win_rate_vs_raw = pairwise_summary["vs_raw_win"]
        hi_metrics.vs_raw_wins = vs_raw_wins
        hi_metrics.vs_raw_total = n_pairs

        hi_metrics.win_rate_vs_generic = pairwise_summary["vs_gen_win"]
        hi_metrics.vs_generic_wins = vs_gen_wins
        hi_metrics.vs_generic_total = n_pairs

        hi_metrics.position_bias_cases_tested = position_bias_tested
        hi_metrics.position_bias_consistent_cases = position_bias_consistent
        hi_metrics.position_bias_consistency_rate = pairwise_summary["position_bias_consistency_rate"]

        # 7. Bootstrap Confidence Intervals (TASK 9)
        ci_results = compute_benchmark_bootstrap_ci(
            cases=cases,
            predictions=by_baseline.get("humanintent", []),
            iterations=self.config.bootstrap_iterations,
            confidence_level=self.config.confidence_level,
            random_seed=self.config.random_seed,
        )
        hi_metrics.confidence_intervals = ci_results

        # 8. Confusion Matrix (TASK 13)
        conf_mat = compute_confusion_matrix(cases, by_baseline.get("humanintent", []))

        # 9. Export All Artifacts (TASK 15 & 21)
        pred_path = self.reporter.export_predictions(all_predictions)
        metrics_path = self.reporter.export_metrics_json(hi_metrics)
        conf_path = self.reporter.export_confusion_matrix(conf_mat)
        err_path = self.reporter.export_error_analysis(cases, by_baseline.get("humanintent", []))
        lat_path = self.reporter.export_latency_json(hi_metrics.stage_latencies)
        call_summary = {
            "avg_llm_calls": hi_metrics.avg_llm_calls,
            "median_calls": hi_metrics.median_calls,
            "p90_calls": hi_metrics.p90_calls,
            "p95_calls": hi_metrics.p95_calls,
            "clear_enhanced_avg_calls": hi_metrics.clear_enhanced_avg_calls,
            "needs_clarification_avg_calls": hi_metrics.needs_clarification_avg_calls,
            "analysis_repair_cases": hi_metrics.analysis_repair_cases,
            "enhancement_repair_cases": hi_metrics.enhancement_repair_cases,
        }
        calls_path = self.reporter.export_call_counts_json(call_summary)
        gates_path = self.reporter.export_release_gates_json(hi_metrics.release_gates)

        report_path = self.reporter.export_markdown_report(
            metrics=hi_metrics,
            baseline_metrics=baseline_metrics,
            pairwise_summary=pairwise_summary,
            confusion_matrix=conf_mat,
            total_benchmark_cases=len(cases),
        )

        logger.info("Evaluation complete! Report exported to: %s", report_path)
        return {
            "run_id": hi_metrics.run_id,
            "metrics_path": metrics_path,
            "report_path": report_path,
            "predictions_path": pred_path,
            "error_analysis_path": err_path,
            "confusion_matrix_path": conf_path,
            "latency_path": lat_path,
            "call_counts_path": calls_path,
            "release_gates_path": gates_path,
            "metrics": hi_metrics.model_dump(),
            "pairwise": pairwise_summary,
        }
