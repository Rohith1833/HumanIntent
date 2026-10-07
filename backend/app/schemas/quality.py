"""Phase 7 Prompt Quality Scoring Schemas."""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class InflationRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class QualityDimension(BaseModel):
    """Evaluation of an individual prompt quality dimension."""

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Dimensional score from 0 to 100.",
    )
    label: str = Field(
        ...,
        description="Qualitative label (e.g., Excellent, Strong, Good, Fair, Needs Improvement).",
    )
    reason: str = Field(
        ...,
        description="Brief, concrete justification for this score.",
    )


class SemanticScoreItem(BaseModel):
    """Individual score item produced by the semantic scorer LLM."""

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score between 0 and 100, restricted to increments of 5.",
    )
    reason: str = Field(
        ...,
        description="Concise rationale referring to the specific rubric.",
    )

    @field_validator("score")
    @classmethod
    def clamp_and_quantize_score(cls, v: int) -> int:
        clamped = max(0, min(100, v))
        # Round to nearest multiple of 5
        remainder = clamped % 5
        if remainder != 0:
            if remainder >= 3:
                clamped = min(100, clamped + (5 - remainder))
            else:
                clamped = max(0, clamped - remainder)
        return clamped


class SemanticDimensionsOutput(BaseModel):
    """Schema for the single structured LLM semantic evaluation call."""

    clarity: SemanticScoreItem
    specificity: SemanticScoreItem
    output_definition: SemanticScoreItem
    structure: SemanticScoreItem
    concision: SemanticScoreItem


class PromptQualityReport(BaseModel):
    """Comprehensive explainable quality report for a validated prompt."""

    overall_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Deterministic weighted aggregate score (0 to 100).",
    )
    overall_label: str = Field(
        ...,
        description="Aggregate performance label (e.g., Excellent, Strong, Good, Fair, Needs Improvement).",
    )
    intent_preservation: QualityDimension
    requirement_coverage: QualityDimension
    constraint_preservation: QualityDimension
    clarity: QualityDimension
    specificity: QualityDimension
    output_definition: QualityDimension
    structure: QualityDimension
    concision: QualityDimension
    inflation_ratio: float = Field(
        ...,
        description="Ratio of enhanced prompt word count to original prompt word count.",
    )
    inflation_risk: InflationRisk = Field(
        ...,
        description="Categorical risk of unnecessary prompt inflation (LOW, MEDIUM, HIGH).",
    )


# Centralized, immutable scoring weights (Total = 1.00 / 100%)
QUALITY_WEIGHTS: dict[str, float] = {
    "intent_preservation": 0.25,
    "requirement_coverage": 0.20,
    "constraint_preservation": 0.15,
    "clarity": 0.15,
    "specificity": 0.10,
    "output_definition": 0.07,
    "structure": 0.05,
    "concision": 0.03,
}


def get_score_label(score: int) -> str:
    """Return polite, standard qualitative label for a 0–100 score."""
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Needs Improvement"
