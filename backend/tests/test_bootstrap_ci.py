"""
Unit tests for Bootstrap statistical estimation and confidence interval correctness (Phase 10).

Validates:
- Point estimate and bootstrap distribution refer to the exact same estimator.
- Metric names cannot become positionally mismatched.
- Synthetic deterministic examples produce exact expected CIs.
- Clarification F1 = 0 cannot accidentally receive another metric's CI.
- Defensive invariants on all primary metrics.
"""

import os
import sys
import pytest

# Ensure repo root and backend are on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkExpected,
    CasePrediction,
    EvaluationBaselineType,
)
from evaluation.bootstrap import compute_bootstrap_ci, compute_benchmark_bootstrap_ci
from evaluation.metrics import compute_run_metrics


def _make_dummy_case(
    case_id: str,
    category: str = "CODING",
    must_preserve: list[str] = None,
    constraints: list[str] = None,
    quantities: list[str] = None,
    must_not_invent: list[str] = None,
    clarification_required: bool = False,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        benchmark_version="test-v1.0",
        category=category,
        difficulty="EASY",
        user_prompt=f"Prompt for {case_id}",
        expected=BenchmarkExpected(
            category=category,
            task_family=category,
            must_preserve=must_preserve or ["python", "docker"],
            constraints=constraints or ["no root"],
            quantities=quantities or ["3 replicas"],
            must_not_invent=must_not_invent or ["kubernetes"],
            clarification_required=clarification_required,
        ),
    )


def _make_dummy_pred(
    case_id: str,
    detected_category: str = "CODING",
    enhanced_prompt: str = "python docker no root 3 replicas",
    clarification_requested: bool = False,
) -> CasePrediction:
    return CasePrediction(
        case_id=case_id,
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="test-model",
        enhanced_prompt=enhanced_prompt,
        clarification_requested=clarification_requested,
        detected_category=detected_category,
        detected_task_family=detected_category,
        quality_score=90.0,
        validation_passed=True,
    )


def test_basic_float_bootstrap_ci():
    """Verify standard float list bootstrap produces valid bounds."""
    sample = [1.0] * 50 + [0.0] * 50
    mean, low, high = compute_bootstrap_ci(sample, iterations=500, random_seed=42)
    assert mean == 0.5
    assert 0.35 <= low <= 0.45
    assert 0.55 <= high <= 0.65
    assert low <= mean <= high


def test_synthetic_deterministic_100_percent():
    """When all cases perfectly preserve every requirement, CI must be strictly [1.0, 1.0]."""
    cases = [_make_dummy_case(f"c_{i}") for i in range(20)]
    preds = [_make_dummy_pred(f"c_{i}") for i in range(20)]

    ci = compute_benchmark_bootstrap_ci(cases, preds, iterations=200, random_seed=42)

    assert ci["category_accuracy"]["point_estimate"] == 1.0
    assert ci["category_accuracy"]["ci_lower"] == 1.0
    assert ci["category_accuracy"]["ci_upper"] == 1.0

    assert ci["requirement_recall"]["point_estimate"] == 1.0
    assert ci["requirement_recall"]["ci_lower"] == 1.0
    assert ci["requirement_recall"]["ci_upper"] == 1.0

    assert ci["constraint_recall"]["point_estimate"] == 1.0
    assert ci["constraint_recall"]["ci_lower"] == 1.0
    assert ci["constraint_recall"]["ci_upper"] == 1.0

    assert ci["quantity_preservation"]["point_estimate"] == 1.0
    assert ci["quantity_preservation"]["ci_lower"] == 1.0
    assert ci["quantity_preservation"]["ci_upper"] == 1.0

    assert ci["unsupported_requirement_rate"]["point_estimate"] == 0.0
    assert ci["unsupported_requirement_rate"]["ci_lower"] == 0.0
    assert ci["unsupported_requirement_rate"]["ci_upper"] == 0.0


def test_clarification_f1_zero_cannot_receive_other_metric_ci():
    """Ensure clarification F1 = 0 cannot accidentally map to an accuracy metric."""
    # 20 cases, 5 require clarification, none requested -> F1 = 0.0
    cases = []
    preds = []
    for i in range(20):
        req_clar = (i < 5)
        cases.append(_make_dummy_case(f"c_{i}", clarification_required=req_clar))
        # 100% correct category, 0% clarification
        preds.append(_make_dummy_pred(f"c_{i}", detected_category="CODING", clarification_requested=False))

    ci = compute_benchmark_bootstrap_ci(cases, preds, iterations=300, random_seed=42)

    # Clarification F1 is 0.0 -> must be strictly [0.0, 0.0]
    assert ci["clarification_f1"]["point_estimate"] == 0.0
    assert ci["clarification_f1"]["mean"] == 0.0
    assert ci["clarification_f1"]["ci_lower"] == 0.0
    assert ci["clarification_f1"]["ci_upper"] == 0.0

    # Category accuracy is 100% -> must NOT be swapped with clarification F1
    assert ci["category_accuracy"]["point_estimate"] == 1.0
    assert ci["category_accuracy"]["ci_lower"] == 1.0
    assert ci["category_accuracy"]["ci_upper"] == 1.0


def test_metric_names_cannot_become_positionally_mismatched():
    """Verify that every metric dictionary key explicitly aligns with compute_run_metrics."""
    cases = []
    preds = []
    for i in range(30):
        # Mix of correct categories, missing constraints, hallucinations
        cat = "CODING" if i % 2 == 0 else "WRITING"
        det_cat = "CODING"
        text = "python" if i % 3 == 0 else "python docker no root 3 replicas kubernetes"
        cases.append(_make_dummy_case(f"c_{i}", category=cat))
        preds.append(_make_dummy_pred(f"c_{i}", detected_category=det_cat, enhanced_prompt=text))

    run_metrics = compute_run_metrics(cases=cases, predictions=preds)
    ci = compute_benchmark_bootstrap_ci(cases, preds, iterations=300, random_seed=42)

    expected_keys = {
        "category_accuracy",
        "task_family_accuracy",
        "requirement_recall",
        "constraint_recall",
        "quantity_preservation",
        "unsupported_requirement_rate",
        "clarification_f1",
    }
    assert set(ci.keys()) == expected_keys

    # Each point estimate in CI output MUST equal the estimator in compute_run_metrics
    assert ci["category_accuracy"]["point_estimate"] == round(run_metrics.category_accuracy, 4)
    assert ci["task_family_accuracy"]["point_estimate"] == round(run_metrics.task_family_accuracy, 4)
    assert ci["requirement_recall"]["point_estimate"] == round(run_metrics.requirement_recall, 4)
    assert ci["constraint_recall"]["point_estimate"] == round(run_metrics.constraint_recall, 4)
    assert ci["quantity_preservation"]["point_estimate"] == round(run_metrics.quantity_preservation, 4)
    assert ci["unsupported_requirement_rate"]["point_estimate"] == round(run_metrics.unsupported_requirement_rate, 4)
    assert ci["clarification_f1"]["point_estimate"] == round(run_metrics.clarification_f1, 4)


def test_point_estimate_and_bootstrap_refer_to_same_estimator():
    """Bootstrap resample mean must closely center on the point estimate."""
    cases = []
    preds = []
    for i in range(40):
        cat = "CODING" if i < 30 else "GENERAL"
        text = "python docker" if i % 2 == 0 else "python"
        cases.append(_make_dummy_case(f"c_{i}", category=cat))
        preds.append(_make_dummy_pred(f"c_{i}", detected_category="CODING", enhanced_prompt=text))

    ci = compute_benchmark_bootstrap_ci(cases, preds, iterations=500, random_seed=42)

    for metric_name, data in ci.items():
        pt = data["point_estimate"]
        mean = data["mean"]
        low = data["ci_lower"]
        high = data["ci_upper"]

        # Mean must be within 0.05 of point estimate for N=40
        assert abs(mean - pt) < 0.05, f"{metric_name}: mean {mean} deviated from point estimate {pt}"
        # Interval must be properly ordered
        assert low <= high, f"{metric_name}: low {low} > high {high}"
        # Point estimate must be bounded within or very close to the confidence interval
        assert low <= pt <= high or abs(pt - low) < 0.02 or abs(pt - high) < 0.02
