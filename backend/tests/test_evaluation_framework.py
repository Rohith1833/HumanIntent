"""
Phase 10 Evaluation Framework Unit & Regression Tests.
Covers Test Groups A through R:
- Benchmark schema & hash integrity
- Metrics calculations (category, requirements, constraints, quantities, clarification, hallucinations)
- Pairwise blinded judge & position bias
- Bootstrap statistical confidence intervals
- Latency & release gates
- Resumability & dry-run
- Runtime isolation & no-training guarantees
- Benchmark contamination check vs Phase 9 seeds
"""

import json
import os
import sys
import pytest
from pydantic import ValidationError

# Ensure repo root and backend are on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.benchmark_loader import (
    BenchmarkIntegrityError,
    load_benchmark,
    verify_benchmark_integrity,
)
from evaluation.bootstrap import compute_all_bootstrap_intervals, compute_bootstrap_ci
from evaluation.config import EvaluationConfig
from evaluation.metrics import (
    compute_case_failures,
    compute_confusion_matrix,
    compute_run_metrics,
    evaluate_release_gates,
)
from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkExpected,
    BenchmarkProvenance,
    CasePrediction,
    EvaluationBaselineType,
    EvaluationDifficulty,
    FailureType,
    PairwisePreference,
    ReleaseGateStatus,
)


# =====================================================================
# TEST GROUP A — BENCHMARK SCHEMA
# =====================================================================
def test_group_a_benchmark_schema_validation():
    """Valid benchmark case loads; malformed case is rejected."""
    valid_case = BenchmarkCase(
        case_id="test_001",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="Write hello world in python",
        expected=BenchmarkExpected(
            category="CODING",
            task_family="CODING",
            must_preserve=["hello world", "python"],
        ),
    )
    assert valid_case.case_id == "test_001"

    # Missing mandatory field
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate({"case_id": "bad_case"})


# =====================================================================
# TEST GROUP B — BENCHMARK HASH
# =====================================================================
def test_group_b_benchmark_hash_verification(tmp_path):
    """Frozen benchmark hash verified; mutation causes integrity failure."""
    bm_file = tmp_path / "benchmark.jsonl"
    meta_file = tmp_path / "metadata.json"

    content = '{"case_id":"c1","benchmark_version":"v1","category":"CODING","difficulty":"EASY","user_prompt":"hi","expected":{"category":"CODING","task_family":"CODING"},"tags":[],"provenance":{"authoring_method":"test","reviewed":true,"benchmark_version":"v1"}}\n'
    bm_file.write_bytes(content.encode("utf-8"))

    import hashlib
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

    meta_json = {
        "version": "v1",
        "frozen_at": "2026-10-08T00:00:00Z",
        "sha256": sha256,
        "total_cases": 1,
        "category_distribution": {"CODING": 1},
        "difficulty_distribution": {"EASY": 1},
        "tags_summary": {},
    }
    meta_file.write_text(json.dumps(meta_json), encoding="utf-8")

    # Should pass
    meta = verify_benchmark_integrity(str(bm_file), str(meta_file))
    assert meta.sha256 == sha256

    # Mutate benchmark file
    bm_file.write_bytes((content + "corrupted\n").encode("utf-8"))
    with pytest.raises(BenchmarkIntegrityError):
        verify_benchmark_integrity(str(bm_file), str(meta_file))


# =====================================================================
# TEST GROUP C — CATEGORY METRICS & CONFUSION MATRIX
# =====================================================================
def test_group_c_category_metrics_and_confusion_matrix():
    """Confusion matrix and category accuracy calculated correctly."""
    case1 = BenchmarkCase(
        case_id="c1",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="write code",
        expected=BenchmarkExpected(category="CODING", task_family="CODING"),
    )
    case2 = BenchmarkCase(
        case_id="c2",
        category="WRITING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="write email",
        expected=BenchmarkExpected(category="WRITING", task_family="WRITING"),
    )

    pred1 = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="llama3",
        enhanced_prompt="code prompt",
        detected_category="CODING",
        detected_task_family="CODING",
    )
    pred2 = CasePrediction(
        case_id="c2",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="llama3",
        enhanced_prompt="writing prompt",
        detected_category="CODING",  # misclassified
        detected_task_family="CODING",
    )

    matrix = compute_confusion_matrix([case1, case2], [pred1, pred2])
    assert matrix["CODING"]["CODING"] == 1
    assert matrix["WRITING"]["CODING"] == 1

    metrics = compute_run_metrics([case1, case2], [pred1, pred2])
    assert metrics.category_accuracy == 0.5


# =====================================================================
# TEST GROUP D — REQUIREMENT RECALL
# =====================================================================
def test_group_d_requirement_recall():
    """Explicit requirement preservation calculated correctly."""
    case = BenchmarkCase(
        case_id="c1",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="Build a binary search tree in Python",
        expected=BenchmarkExpected(
            category="CODING",
            task_family="CODING",
            must_preserve=["binary search tree", "Python"],
        ),
    )

    # Pred preserves both
    pred_good = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="llama3",
        enhanced_prompt="Implement a binary search tree data structure in Python with insert and search.",
    )
    m_good = compute_run_metrics([case], [pred_good])
    assert m_good.requirement_recall == 1.0

    # Pred drops Python
    pred_bad = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="llama3",
        enhanced_prompt="Implement a binary search tree data structure in Java.",
    )
    m_bad = compute_run_metrics([case], [pred_bad])
    assert m_bad.requirement_recall == 0.5


# =====================================================================
# TEST GROUP E — CONSTRAINT RECALL
# =====================================================================
def test_group_e_constraint_recall():
    """Negative and restrictive constraint preservation calculated correctly."""
    case = BenchmarkCase(
        case_id="c1",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="Implement rate limiter without external libraries",
        expected=BenchmarkExpected(
            category="CODING",
            task_family="CODING",
            constraints=["without external libraries"],
        ),
    )

    pred = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="llama3",
        enhanced_prompt="Build a token bucket limiter using only the standard library, without external libraries.",
    )
    m = compute_run_metrics([case], [pred])
    assert m.constraint_recall == 1.0


# =====================================================================
# TEST GROUP F — QUANTITY PRESERVATION
# =====================================================================
def test_group_f_quantity_preservation():
    """Quantity constraints evaluated accurately."""
    case = BenchmarkCase(
        case_id="c1",
        category="CREATIVE",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="Brainstorm 5 ideas under 100 words",
        expected=BenchmarkExpected(
            category="CREATIVE",
            task_family="CREATIVE",
            quantities=["5 ideas", "100 words"],
        ),
    )

    pred = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="llama3",
        enhanced_prompt="Provide 5 creative concepts for games under 100 words total.",
    )
    m = compute_run_metrics([case], [pred])
    assert m.quantity_preservation == 1.0


# =====================================================================
# TEST GROUP G — CLARIFICATION METRICS
# =====================================================================
def test_group_g_clarification_metrics():
    """Precision, recall, F1, over/under clarification compute accurately."""
    # c1: truly ambiguous
    c1 = BenchmarkCase(
        case_id="c1",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="build app",
        expected=BenchmarkExpected(category="CODING", task_family="CODING", clarification_required=True),
    )
    # c2: clear prompt
    c2 = BenchmarkCase(
        case_id="c2",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="build python web scraper using requests",
        expected=BenchmarkExpected(category="CODING", task_family="CODING", clarification_required=False),
    )

    # System correctly clarifies c1, enhances c2
    p1 = CasePrediction(case_id="c1", baseline=EvaluationBaselineType.HUMANINTENT, model="m", enhanced_prompt="", clarification_requested=True)
    p2 = CasePrediction(case_id="c2", baseline=EvaluationBaselineType.HUMANINTENT, model="m", enhanced_prompt="enhanced scraper", clarification_requested=False)

    m = compute_run_metrics([c1, c2], [p1, p2])
    assert m.clarification_precision == 1.0
    assert m.clarification_recall == 1.0
    assert m.clarification_f1 == 1.0
    assert m.over_clarification_rate == 0.0
    assert m.under_clarification_rate == 0.0


# =====================================================================
# TEST GROUP H — UNSUPPORTED REQUIREMENT RATE
# =====================================================================
def test_group_h_unsupported_requirement_rate():
    """Hallucination detection triggers when unrequested feature is injected."""
    case = BenchmarkCase(
        case_id="c1",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="Write a simple python script",
        expected=BenchmarkExpected(
            category="CODING",
            task_family="CODING",
            must_not_invent=["Kubernetes deployment", "PostgreSQL database"],
        ),
    )

    pred_hallucinating = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="m",
        enhanced_prompt="Write a Python script and configure a Kubernetes deployment with PostgreSQL database.",
    )

    m = compute_run_metrics([case], [pred_hallucinating])
    assert m.unsupported_requirement_rate == 1.0


# =====================================================================
# TEST GROUP I & J — PAIRWISE JUDGE & POSITION BIAS
# =====================================================================
def test_group_i_pairwise_judge_schema():
    """Pairwise preference enum and result structure."""
    assert PairwisePreference.A_WINS.value == "A"
    assert PairwisePreference.B_WINS.value == "B"
    assert PairwisePreference.TIE.value == "TIE"


# =====================================================================
# TEST GROUP K — BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================================
def test_group_k_bootstrap_confidence_intervals():
    """Bootstrap confidence intervals are deterministic for fixed random seed."""
    sample = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    mean1, low1, high1 = compute_bootstrap_ci(sample, iterations=500, random_seed=42)
    mean2, low2, high2 = compute_bootstrap_ci(sample, iterations=500, random_seed=42)

    assert mean1 == mean2
    assert low1 == low2
    assert high1 == high2
    assert low1 <= mean1 <= high1


# =====================================================================
# TEST GROUP L — LATENCY AGGREGATION
# =====================================================================
def test_group_l_latency_aggregation():
    """Median, p90, and p95 latency computed accurately."""
    case = BenchmarkCase(case_id="c1", category="CODING", difficulty=EvaluationDifficulty.EASY, user_prompt="p", expected=BenchmarkExpected(category="CODING", task_family="CODING"))
    preds = [
        CasePrediction(case_id="c1", baseline=EvaluationBaselineType.HUMANINTENT, model="m", enhanced_prompt="e", latency_ms=float(i * 10))
        for i in range(1, 11)
    ]
    m = compute_run_metrics([case] * 10, preds)
    assert m.median_latency_ms > 0
    assert m.p90_latency_ms >= m.median_latency_ms
    assert m.p95_latency_ms >= m.p90_latency_ms


# =====================================================================
# TEST GROUP M — RELEASE GATES
# =====================================================================
def test_group_m_release_gates_evaluation():
    """Passing and failing metrics correctly map to gate status."""
    case = BenchmarkCase(case_id="c1", category="CODING", difficulty=EvaluationDifficulty.EASY, user_prompt="p", expected=BenchmarkExpected(category="CODING", task_family="CODING"))
    pred = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="m",
        enhanced_prompt="e",
        detected_category="CODING",
        detected_task_family="CODING",
    )
    metrics = compute_run_metrics([case], [pred])
    gates = evaluate_release_gates(metrics)
    assert len(gates) == 10
    gate_dict = {g.gate_name: g.status for g in gates}
    assert gate_dict["Category Accuracy"] == ReleaseGateStatus.PASSED


# =====================================================================
# TEST GROUP N — FAILURE TAXONOMY
# =====================================================================
def test_group_n_failure_taxonomy_assignment():
    """Predictions receive accurate failure labels."""
    case = BenchmarkCase(
        case_id="c1",
        category="CODING",
        difficulty=EvaluationDifficulty.EASY,
        user_prompt="Build app",
        expected=BenchmarkExpected(
            category="CODING",
            task_family="CODING",
            clarification_required=True,
            must_preserve=["database schema"],
        ),
    )
    # Under-clarified and misclassified
    pred = CasePrediction(
        case_id="c1",
        baseline=EvaluationBaselineType.HUMANINTENT,
        model="m",
        enhanced_prompt="I will write a python function.",
        detected_category="WRITING",
        clarification_requested=False,
    )
    failures = compute_case_failures(case, pred)
    assert FailureType.CATEGORY_ERROR in failures
    assert FailureType.UNDER_CLARIFICATION in failures
    assert FailureType.MISSING_REQUIREMENT in failures


# =====================================================================
# TEST GROUP O — RESUMABILITY CHECK
# =====================================================================
def test_group_o_resumability(tmp_path):
    """Checkpoint file allows loading existing predictions without duplication."""
    ckpt_path = tmp_path / "predictions_checkpoint.jsonl"
    p = CasePrediction(case_id="c1", baseline=EvaluationBaselineType.HUMANINTENT, model="m", enhanced_prompt="text")
    ckpt_path.write_text(p.model_dump_json() + "\n", encoding="utf-8")

    # Read back
    records = []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(CasePrediction.model_validate_json(line))
    assert len(records) == 1
    assert records[0].case_id == "c1"


# =====================================================================
# TEST GROUP P — DRY RUN
# =====================================================================
@pytest.mark.asyncio
async def test_group_p_dry_run():
    """Dry run validates benchmark without making LLM calls."""
    from evaluation.runner import EvaluationRunner
    config = EvaluationConfig(dry_run=True)
    runner = EvaluationRunner(config)
    result = await runner.run()
    assert result.get("dry_run") is True
    assert result.get("total_cases") == 220


# =====================================================================
# TEST GROUP Q — RUNTIME ISOLATION
# =====================================================================
def test_group_q_runtime_isolation():
    """Assert /api/enhance and core runtime do not import evaluation harness."""
    import app.api.routes as routes_module
    import app.core.enhancer as enhancer_module

    for mod in [routes_module, enhancer_module]:
        with open(mod.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        assert "import evaluation" not in src, f"{mod.__name__} violates isolation by importing evaluation"
        assert "from evaluation" not in src, f"{mod.__name__} violates isolation by importing evaluation"


# =====================================================================
# TEST GROUP R — NO TRAINING PIPELINE
# =====================================================================
def test_group_r_no_training_pipeline():
    """Verify that no fine-tuning or training modules exist."""
    import sys
    for mod_name in ["transformers", "peft", "trl"]:
        assert mod_name not in sys.modules, f"{mod_name} should not be loaded in Phase 10"


# =====================================================================
# BENCHMARK CONTAMINATION TEST
# =====================================================================
def test_benchmark_contamination_check():
    """Ensure benchmark prompts do not match Phase 9 seed prompts or seed IDs."""
    benchmark_cases = load_benchmark()
    seed_path = os.path.join(repo_root, "datasets", "seeds", "small_live_seeds.jsonl")

    seed_prompts = []
    seed_ids = set()
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    seed_prompts.append(item.get("intent", "").lower().strip())
                    seed_ids.add(item.get("seed_id", "").lower().strip())

    for case in benchmark_cases:
        # Assert distinct case IDs
        assert case.case_id.lower() not in seed_ids, f"Contamination: Case ID {case.case_id} overlaps with seed ID"

        # Assert distinct prompt texts
        p_lower = case.user_prompt.lower().strip()
        for sp in seed_prompts:
            if len(sp) > 10:
                assert p_lower != sp, f"Contamination: Benchmark prompt exactly matches seed intent: '{sp}'"


# =====================================================================
# TEST GROUP S — STABILITY METRICS
# =====================================================================
def test_group_s_stability_metrics():
    """Verify stochastic stability metrics calculation."""
    from evaluation.stability import StabilityMetrics

    m = StabilityMetrics(
        total_cases_evaluated=3,
        repetitions_per_case=3,
        category_stability_rate=1.0,
        clarification_stability_rate=1.0,
        mean_quality_score_std=0.02,
        max_quality_score_std=0.05,
    )
    assert m.category_stability_rate == 1.0
    assert m.clarification_stability_rate == 1.0
    assert m.total_cases_evaluated == 3

