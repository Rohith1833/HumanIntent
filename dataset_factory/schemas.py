from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Origin of a dataset candidate."""
    SYNTHETIC = "SYNTHETIC"
    CONSENTED_REAL = "CONSENTED_REAL"


class VariantType(str, Enum):
    """Target phrasing mode for human-like variation."""
    CLEAN = "CLEAN"
    CASUAL = "CASUAL"
    MESSY = "MESSY"
    SHORT = "SHORT"
    COMPOUND = "COMPOUND"
    SELF_CORRECTING = "SELF_CORRECTING"
    CONSTRAINED = "CONSTRAINED"
    ALREADY_GOOD = "ALREADY_GOOD"
    AMBIGUOUS = "AMBIGUOUS"


class CurationStatus(str, Enum):
    """Curation lifecycle status of a candidate record."""
    RAW = "RAW"
    ELIGIBLE = "ELIGIBLE"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class CurationIssueType(str, Enum):
    """Standardized issue types identified by the Dataset Curation Gate."""
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    MISSING_CONSTRAINT = "MISSING_CONSTRAINT"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    SOURCE_MISSING = "SOURCE_MISSING"
    TASK_FAMILY_DRIFT = "TASK_FAMILY_DRIFT"
    UNSUPPORTED_REQUIREMENT = "UNSUPPORTED_REQUIREMENT"
    MALFORMED_TARGET = "MALFORMED_TARGET"
    PLACEHOLDER_CONTENT = "PLACEHOLDER_CONTENT"
    COMPOUND_TASK_LOSS = "COMPOUND_TASK_LOSS"
    EXCESSIVE_INFLATION = "EXCESSIVE_INFLATION"
    META_CONTENT = "META_CONTENT"
    OTHER = "OTHER"


class RejectionReason(str, Enum):
    """Standardized reasons for candidate rejection."""
    NO_CONSENT = "NO_CONSENT"
    PRIVACY_RISK = "PRIVACY_RISK"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    LOW_QUALITY = "LOW_QUALITY"
    INTENT_DRIFT = "INTENT_DRIFT"
    DUPLICATE = "DUPLICATE"
    MALFORMED = "MALFORMED"
    PROMPT_INFLATION = "PROMPT_INFLATION"
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    MISSING_CONSTRAINT = "MISSING_CONSTRAINT"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    SOURCE_MISSING = "SOURCE_MISSING"
    TASK_FAMILY_DRIFT = "TASK_FAMILY_DRIFT"
    UNSUPPORTED_REQUIREMENT = "UNSUPPORTED_REQUIREMENT"
    MALFORMED_TARGET = "MALFORMED_TARGET"
    PLACEHOLDER_CONTENT = "PLACEHOLDER_CONTENT"
    COMPOUND_TASK_LOSS = "COMPOUND_TASK_LOSS"
    META_CONTENT = "META_CONTENT"
    OTHER = "OTHER"


class DatasetCurationIssue(BaseModel):
    """Specific flaw detected in a candidate target prompt during offline curation."""
    type: CurationIssueType
    severity: str = "ERROR"  # "ERROR", "WARNING", "INFO"
    message: str


class DatasetCurationReport(BaseModel):
    """Detailed invariant audit produced by the offline Dataset Curation Gate."""
    is_acceptable: bool = True
    requirement_recall_ok: bool = True
    constraint_recall_ok: bool = True
    quantity_recall_ok: bool = True
    source_dependency_ok: bool = True
    task_family_ok: bool = True
    target_well_formed: bool = True
    no_placeholder_content: bool = True
    no_internal_meta_content: bool = True
    no_unsupported_requirement: bool = True
    issues: List[DatasetCurationIssue] = Field(default_factory=list)


class SeedIntent(BaseModel):
    """Manually curated semantic seed intent describing a core user task."""
    seed_id: str = Field(..., description="Unique seed identifier (e.g. coding_001)")
    category: str = Field(..., description="Canonical category: CODING, DEBUGGING, WRITING, etc.")
    intent: str = Field(..., description="High-level description of what the user wants to accomplish")
    core_requirements: List[str] = Field(default_factory=list, description="Must-have functional requirements")
    optional_constraints: List[str] = Field(default_factory=list, description="Optional constraints or preferences")
    difficulty: str = Field("medium", description="simple, medium, hard, complex")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags for filtering and coverage")
    already_good_reference: Optional[str] = Field(None, description="Reference exemplary prompt if applicable")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid_cats = {
            "CODING", "DEBUGGING", "WRITING", "EDUCATION", "RESEARCH",
            "PLANNING", "DATA_ANALYSIS", "CREATIVE", "CREATIVE_IDEATION",
            "IMAGE_GENERATION", "SUMMARIZATION", "GENERAL"
        }
        upper = v.upper().strip()
        if upper == "CREATIVE_IDEATION":
            upper = "CREATIVE"
        if upper not in valid_cats:
            raise ValueError(f"Invalid category '{v}'. Must be one of {sorted(list(valid_cats))}")
        return upper


class SyntheticVariant(BaseModel):
    """A generated human-like prompt variant derived from a SeedIntent."""
    variant_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    seed_id: str
    prompt: str
    variant_type: VariantType
    intended_requirements: List[str] = Field(default_factory=list)
    intent_preserved: bool = True


class NormalizedIntent(BaseModel):
    """Structured decomposition of intent extracted by the teacher analyzer."""
    objective: str
    explicit_requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    missing_context: List[str] = Field(default_factory=list)


class QualityMetadata(BaseModel):
    """Quality and validation evaluation metrics."""
    validation_passed: bool
    quality_score: float = 0.0
    inflation_ratio: Optional[float] = None
    validation_issues: List[str] = Field(default_factory=list)


class ProvenanceMetadata(BaseModel):
    """Full traceability metadata for dataset reproducibility."""
    seed_id: Optional[str] = None
    variant_id: Optional[str] = None
    source_id: Optional[str] = None  # One-way hashed session id if real
    generation_model: Optional[str] = None
    teacher_pipeline_version: str = "humanintent-v1.0"
    validator_version: str = "v1.0"
    dataset_version: str


class PrivacyMetadata(BaseModel):
    """Privacy sanitation tracking."""
    sanitized: bool = False
    pii_types_removed: List[str] = Field(default_factory=list)


class RecordLabels(BaseModel):
    """Behavioral and semantic label tags."""
    already_good: bool = False
    compound_request: bool = False
    self_correction: bool = False
    ambiguous: bool = False
    negative_constraint: bool = False
    quantitative_constraint: bool = False


class DatasetRecord(BaseModel):
    """Canonical HumanIntent dataset candidate record."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_version: str
    source_type: SourceType
    source_id: Optional[str] = None
    category: str
    original_prompt: str
    normalized_intent: Optional[NormalizedIntent] = None
    enhanced_prompt: Optional[str] = None
    clarification_required: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    quality: QualityMetadata
    provenance: ProvenanceMetadata
    privacy: PrivacyMetadata = Field(default_factory=PrivacyMetadata)
    labels: RecordLabels = Field(default_factory=RecordLabels)
    curation_status: CurationStatus = CurationStatus.RAW
    rejection_reasons: List[RejectionReason] = Field(default_factory=list)
    curation_report: Optional[DatasetCurationReport] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Training Export Formats

class SFTMessage(BaseModel):
    role: str
    content: str


class SFTExample(BaseModel):
    """Supervised Fine-Tuning pair formatted as chat messages."""
    messages: List[SFTMessage]
    metadata: Dict[str, Any]


class ClarificationExample(BaseModel):
    """Ambiguity / clarification pair."""
    messages: List[SFTMessage]
    metadata: Dict[str, Any]


class PreferenceExample(BaseModel):
    """Chosen vs. Rejected prompt optimization pair."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any]


class DatasetManifest(BaseModel):
    """Complete build manifest for a versioned dataset."""
    version: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    factory_version: str = "0.1.0"
    random_seed: int = 42
    generator_model: str = "llama3:latest"
    record_counts: Dict[str, int] = Field(default_factory=dict)
    category_counts: Dict[str, int] = Field(default_factory=dict)
    source_counts: Dict[str, int] = Field(default_factory=dict)
    variant_counts: Dict[str, int] = Field(default_factory=dict)
    excluded_counts: Dict[str, int] = Field(default_factory=dict)
    filter_reasons: Dict[str, int] = Field(default_factory=dict)
    curation_rejection_counts: Dict[str, int] = Field(default_factory=dict)
    target_integrity_pass_rate: float = 0.0
    source_dependency_rejections: int = 0
    constraint_recall_failures: int = 0
    malformed_target_rejections: int = 0
    quality_distribution: Dict[str, Any] = Field(default_factory=dict)
    consent_count: int = 0
    sanitization_count: int = 0
    dedup_count: int = 0
    split_counts: Dict[str, int] = Field(default_factory=dict)
    generation_parameters: Dict[str, Any] = Field(default_factory=dict)
    filter_configuration: Dict[str, Any] = Field(default_factory=dict)
