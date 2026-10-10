"""
Phase 10 Formal Evaluation & Benchmarking Schemas.
Defines invariant-based benchmark cases, evaluation predictions, judge results,
metrics, and failure taxonomy.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationDifficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class BenchmarkExpected(BaseModel):
    """
    Invariant-based ground truth specifications.
    Does NOT specify an exact string target, but rather semantic invariants.
    """
    category: str
    task_family: str
    objective_concepts: List[str] = Field(default_factory=list, description="Core objective semantic tokens")
    must_preserve: List[str] = Field(default_factory=list, description="Explicit requirements that must survive")
    must_not_invent: List[str] = Field(default_factory=list, description="Hallucination traps or unrequested features")
    constraints: List[str] = Field(default_factory=list, description="Negative or restrictive constraints")
    quantities: List[str] = Field(default_factory=list, description="Numeric and unit bounds")
    clarification_required: bool = False
    clarification_dimensions: List[str] = Field(default_factory=list, description="Dimensions requiring clarification")
    source_required: bool = False
    already_good: bool = False


class BenchmarkProvenance(BaseModel):
    """Metadata regarding the authoring and review of a benchmark case."""
    authoring_method: str = "curated_frozen"
    reviewed: bool = True
    benchmark_version: str = "humanintent-benchmark-v1.0"


class BenchmarkCase(BaseModel):
    """Authoritative frozen benchmark test case."""
    case_id: str
    benchmark_version: str = "humanintent-benchmark-v1.0"
    category: str
    difficulty: EvaluationDifficulty
    user_prompt: str
    expected: BenchmarkExpected
    tags: List[str] = Field(default_factory=list)
    provenance: BenchmarkProvenance = Field(default_factory=BenchmarkProvenance)


class BenchmarkMetadata(BaseModel):
    """Metadata and tamper-evident SHA-256 digest for frozen benchmark."""
    version: str
    frozen_at: str
    sha256: str
    total_cases: int
    category_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    tags_summary: Dict[str, int]


class EvaluationBaselineType(str, Enum):
    RAW = "raw"
    GENERIC = "generic"
    NO_VALIDATOR_ABLATION = "no_validator_ablation"
    HUMANINTENT = "humanintent"


class CasePrediction(BaseModel):
    """Output recorded for a single benchmark case under a specific baseline."""
    case_id: str
    baseline: EvaluationBaselineType
    model: str
    enhanced_prompt: str
    clarification_requested: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    detected_category: Optional[str] = None
    detected_task_family: Optional[str] = None
    quality_score: Optional[float] = None
    validation_passed: bool = True
    latency_ms: float = 0.0
    llm_call_count: int = 1

    # Granular call accounting per case (TASK 5)
    analysis_calls: int = 1
    enhancement_calls: int = 0
    validation_calls: int = 0
    scoring_calls: int = 0
    analysis_repair_calls: int = 0
    enhancement_repair_calls: int = 0
    total_llm_calls: int = 1

    # Stage latencies per case in milliseconds (TASK 7)
    analysis_latency_ms: float = 0.0
    enhancement_latency_ms: float = 0.0
    validation_latency_ms: float = 0.0
    scoring_latency_ms: float = 0.0
    repair_latency_ms: float = 0.0

    raw_response: Optional[Dict[str, Any]] = None


class FailureType(str, Enum):
    CATEGORY_ERROR = "CATEGORY_ERROR"
    TASK_FAMILY_ERROR = "TASK_FAMILY_ERROR"
    OBJECTIVE_DRIFT = "OBJECTIVE_DRIFT"
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    MISSING_CONSTRAINT = "MISSING_CONSTRAINT"
    QUANTITY_ERROR = "QUANTITY_ERROR"
    INVENTED_REQUIREMENT = "INVENTED_REQUIREMENT"
    OVER_CLARIFICATION = "OVER_CLARIFICATION"
    UNDER_CLARIFICATION = "UNDER_CLARIFICATION"
    SOURCE_DEPENDENCY_ERROR = "SOURCE_DEPENDENCY_ERROR"
    ANALYSIS_FAILURE = "ANALYSIS_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    META_LEAKAGE = "META_LEAKAGE"
    EXCESSIVE_INFLATION = "EXCESSIVE_INFLATION"
    OTHER = "OTHER"


class PairwisePreference(str, Enum):
    A_WINS = "A"
    B_WINS = "B"
    TIE = "TIE"


class SemanticJudgeResult(BaseModel):
    """Independent semantic judge evaluation of candidate enhancement."""
    intent_preserved: bool = True
    objective_drift: bool = False
    unsupported_requirements: List[str] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)
    constraint_violations: List[str] = Field(default_factory=list)
    quantity_violations: List[str] = Field(default_factory=list)
    quality_preference: str = "TIE"  # "RAW", "ENHANCED", "TIE"
    explanation_summary: str = ""


class PairwiseJudgeResult(BaseModel):
    """Blinded pairwise comparison result."""
    chosen_label: PairwisePreference  # "A", "B", or "TIE"
    decoded_winner: str  # "humanintent", baseline_name, or "tie"
    reasoning: str = ""
    candidate_a_baseline: str
    candidate_b_baseline: str
    swap_consistent: Optional[bool] = None


class ReleaseGateStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class ReleaseGateResult(BaseModel):
    gate_name: str
    target_threshold: str
    actual_value: float
    status: ReleaseGateStatus
    description: str


class BenchmarkRunMetrics(BaseModel):
    """Comprehensive aggregated metrics for an evaluation run."""
    run_id: str
    benchmark_version: str
    model: str
    total_cases: int

    # Primary accuracy & preservation metrics
    category_accuracy: float = 0.0
    task_family_accuracy: float = 0.0
    requirement_recall: float = 0.0
    constraint_recall: float = 0.0
    quantity_preservation: float = 0.0
    unsupported_requirement_rate: float = 0.0
    objective_drift_rate: float = 0.0

    # Clarification metrics
    clarification_precision: float = 0.0
    clarification_recall: float = 0.0
    clarification_f1: float = 0.0
    over_clarification_rate: float = 0.0
    under_clarification_rate: float = 0.0
    source_dependency_accuracy: float = 0.0

    # System metrics
    meta_leakage_rate: float = 0.0
    already_good_preservation_rate: float = 0.0
    median_expansion_ratio: float = 1.0
    p90_expansion_ratio: float = 1.0
    p95_expansion_ratio: float = 1.0

    # Validation & repair rates (TASK 8)
    validation_first_pass_success: float = 0.0
    analysis_repair_rate: float = 0.0
    analysis_repair_success_rate: float = 0.0
    enhancement_repair_rate: float = 0.0
    enhancement_repair_success_rate: float = 0.0

    # Latency & Calls summary (TASK 6)
    median_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_llm_calls: float = 1.0
    median_calls: float = 1.0
    p90_calls: float = 1.0
    p95_calls: float = 1.0

    # Call budget breakdown (TASK 6)
    clear_enhanced_avg_calls: float = 0.0
    needs_clarification_avg_calls: float = 0.0
    analysis_repair_cases: int = 0
    enhancement_repair_cases: int = 0

    # Stage Latency distribution (TASK 7)
    stage_latencies: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Pairwise Win Rates vs Baselines & Denominators (TASK 16 & 17)
    win_rate_vs_raw: float = 0.0
    vs_raw_wins: int = 0
    vs_raw_total: int = 0
    win_rate_vs_generic: float = 0.0
    vs_generic_wins: int = 0
    vs_generic_total: int = 0

    # Position bias measurement (TASK 19)
    position_bias_cases_tested: int = 0
    position_bias_consistent_cases: int = 0
    position_bias_consistency_rate: float = 1.0

    # Release Gates & Breakdown
    release_gates: List[ReleaseGateResult] = Field(default_factory=list)
    failure_counts: Dict[str, int] = Field(default_factory=dict)
    confidence_intervals: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    category_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    difficulty_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
