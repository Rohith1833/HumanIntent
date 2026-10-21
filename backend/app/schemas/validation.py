from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ValidationSeverity(str, Enum):
    """Severity levels for detected intent-preservation issues."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationIssueType(str, Enum):
    """Categorical types of intent-preservation failures."""

    OBJECTIVE_DRIFT = "OBJECTIVE_DRIFT"
    TASK_FAMILY_DRIFT = "TASK_FAMILY_DRIFT"
    ANALYSIS_OBJECTIVE_DRIFT = "ANALYSIS_OBJECTIVE_DRIFT"
    UNSUPPORTED_ANALYSIS_REQUIREMENT = "UNSUPPORTED_ANALYSIS_REQUIREMENT"
    UNSUPPORTED_ANALYSIS_CONSTRAINT = "UNSUPPORTED_ANALYSIS_CONSTRAINT"
    UNSUPPORTED_ANALYSIS_PREFERENCE = "UNSUPPORTED_ANALYSIS_PREFERENCE"
    ANALYSIS_GROUNDING_FAILURE = "ANALYSIS_GROUNDING_FAILURE"
    SEMANTIC_GROUNDING_MISMATCH = "SEMANTIC_GROUNDING_MISMATCH"
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    MISSING_CONSTRAINT = "MISSING_CONSTRAINT"
    MISSING_PREFERENCE = "MISSING_PREFERENCE"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    INVENTED_REQUIREMENT = "INVENTED_REQUIREMENT"
    INVENTED_TECHNOLOGY = "INVENTED_TECHNOLOGY"
    INVENTED_FEATURE = "INVENTED_FEATURE"
    CONSTRAINT_CONTRADICTION = "CONSTRAINT_CONTRADICTION"
    REJECTED_OPTION_REINTRODUCED = "REJECTED_OPTION_REINTRODUCED"
    MISSING_CONTEXT_PROMOTED = "MISSING_CONTEXT_PROMOTED"
    COMPOUND_TASK_LOSS = "COMPOUND_TASK_LOSS"
    SOURCE_REQUIREMENT_DROPPED = "SOURCE_REQUIREMENT_DROPPED"
    ANALYSIS_INCOMPLETE = "ANALYSIS_INCOMPLETE"
    SOURCE_COVERAGE_FAILED = "SOURCE_COVERAGE_FAILED"
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
    SOURCE_CHUNK_EXTRACTION_FAILED = "SOURCE_CHUNK_EXTRACTION_FAILED"
    SOURCE_CHUNK_COVERAGE_FAILED = "SOURCE_CHUNK_COVERAGE_FAILED"
    GLOBAL_RECONCILIATION_FAILED = "GLOBAL_RECONCILIATION_FAILED"
    CLARIFICATION_MISMATCH = "CLARIFICATION_MISMATCH"
    META_COMMENTARY = "META_COMMENTARY"
    OTHER = "OTHER"


class ValidationIssue(BaseModel):
    """A single detected intent-preservation issue."""

    type: ValidationIssueType = Field(
        ...,
        description="The category of intent-preservation issue.",
    )
    severity: ValidationSeverity = Field(
        ...,
        description="Issue severity: ERROR (invalidates enhancement), WARNING (potential risk), INFO (observation).",
    )
    message: str = Field(
        ...,
        description="Clear, concise description of the discrepancy.",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Excerpt from user request, analysis, or enhanced prompt demonstrating the discrepancy.",
    )


class PromptValidation(BaseModel):
    """Result of validating an enhanced prompt against the user's intent."""

    is_valid: bool = Field(
        ...,
        description="True if no ERROR severity issues exist; False if one or more ERROR issues exist.",
    )
    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="Structured list of detected validation issues.",
    )

    @model_validator(mode="after")
    def enforce_error_validity_consistency(self) -> "PromptValidation":
        """Consistency rule: If any issue is an ERROR, is_valid must be False."""
        has_error = any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
        if has_error:
            self.is_valid = False
        return self
