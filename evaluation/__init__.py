"""Phase 10 Evaluation Package."""

from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkExpected,
    CasePrediction,
    SemanticJudgeResult,
    PairwiseJudgeResult,
    ReleaseGateResult,
    BenchmarkRunMetrics,
    FailureType,
    EvaluationBaselineType,
)
from evaluation.config import EvaluationConfig
from evaluation.benchmark_loader import load_benchmark, verify_benchmark_integrity
from evaluation.runner import EvaluationRunner
from evaluation.metrics import compute_run_metrics, evaluate_release_gates
from evaluation.bootstrap import compute_bootstrap_ci, compute_benchmark_bootstrap_ci
from evaluation.stability import StabilityMetrics, run_stability_evaluation

__all__ = [
    "BenchmarkCase",
    "BenchmarkExpected",
    "CasePrediction",
    "SemanticJudgeResult",
    "PairwiseJudgeResult",
    "ReleaseGateResult",
    "BenchmarkRunMetrics",
    "FailureType",
    "EvaluationBaselineType",
    "EvaluationConfig",
    "load_benchmark",
    "verify_benchmark_integrity",
    "EvaluationRunner",
    "compute_run_metrics",
    "evaluate_release_gates",
    "compute_bootstrap_ci",
    "compute_benchmark_bootstrap_ci",
    "StabilityMetrics",
    "run_stability_evaluation",
]
