from typing import Annotated, Literal, Optional, Union
from uuid import UUID
from pydantic import BaseModel, Field, field_validator
from app.schemas.analysis import ClarificationAnswer, ClarificationQuestion, PromptAnalysis
from app.schemas.quality import PromptQualityReport
from app.schemas.telemetry import StageTimingDiagnostic
from app.schemas.validation import PromptValidation


class EnhanceRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The rough human prompt to enhance.",
    )
    session_id: Optional[UUID] = Field(
        default=None,
        description="Optional persistent session identifier for multi-round journeys.",
    )
    clarification_answers: Optional[list[ClarificationAnswer]] = Field(
        default=None,
        description="User answers to previously asked clarification questions.",
    )
    allow_reasonable_assumptions: bool = Field(
        default=False,
        description="If True, bypasses clarification and instructs the enhancer to proceed with safe, minimal assumptions.",
    )
    mode: Optional[Literal["quick", "professional", "deep"]] = Field(
        default=None,
        description="Enhancement mode: 'quick' (fastest single-call), 'professional' (balanced production fast path), 'deep' (comprehensive multi-aspect)",
    )

    @field_validator("prompt")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return stripped


class EnhancedResponse(BaseModel):
    status: Literal["enhanced"] = "enhanced"
    session_id: UUID
    original_prompt: str
    analysis: PromptAnalysis
    enhanced_prompt: str
    model: str
    template: Optional[str] = None
    validation: Optional[PromptValidation] = None
    quality: Optional[PromptQualityReport] = None
    processing_mode: str = Field(default="normal", description="Execution mode: 'normal' or 'long_input'.")
    mode: str = Field(default="professional", description="Enhancement mode requested: 'quick', 'professional', or 'deep'.")
    diagnostic_codes: list[str] = Field(default_factory=list, description="Diagnostic observation codes.")
    stage_timings: list[StageTimingDiagnostic] = Field(default_factory=list, description="Per-stage latency and token telemetry.")
    total_ms: Optional[float] = Field(default=None, description="Total measured end-to-end latency in milliseconds.")
    inference_ms: Optional[float] = Field(default=None, description="Measured model inference latency in milliseconds.")
    history_saved: bool = True


class ClarificationRequiredResponse(BaseModel):
    status: Literal["needs_clarification"] = "needs_clarification"
    session_id: UUID
    original_prompt: str
    analysis: PromptAnalysis
    questions: list[ClarificationQuestion]
    reason_codes: list[str] = Field(default_factory=list, description="Structured machine-readable reason codes.")
    processing_mode: str = Field(default="normal", description="Execution mode: 'normal' or 'long_input'.")
    mode: str = Field(default="professional", description="Enhancement mode requested: 'quick', 'professional', or 'deep'.")
    diagnostic_codes: list[str] = Field(default_factory=list, description="Diagnostic observation codes.")
    stage_timings: list[StageTimingDiagnostic] = Field(default_factory=list, description="Per-stage latency and token telemetry.")
    total_ms: Optional[float] = Field(default=None, description="Total measured end-to-end latency in milliseconds.")
    model: str
    history_saved: bool = True


class ValidationFailedResponse(BaseModel):
    status: Literal["validation_failed"] = "validation_failed"
    session_id: UUID
    original_prompt: str
    analysis: PromptAnalysis
    validation: PromptValidation
    reason_codes: list[str] = Field(default_factory=list, description="Structured machine-readable reason codes.")
    processing_mode: str = Field(default="normal", description="Execution mode: 'normal' or 'long_input'.")
    mode: str = Field(default="professional", description="Enhancement mode requested: 'quick', 'professional', or 'deep'.")
    diagnostic_codes: list[str] = Field(default_factory=list, description="Diagnostic observation codes.")
    stage_timings: list[StageTimingDiagnostic] = Field(default_factory=list, description="Per-stage latency and token telemetry.")
    total_ms: Optional[float] = Field(default=None, description="Total measured end-to-end latency in milliseconds.")
    message: str = "HumanIntent could not produce a sufficiently faithful enhanced prompt after one repair attempt."
    model: str
    history_saved: bool = True


class AnalysisFailedResponse(BaseModel):
    status: Literal["analysis_failed"] = "analysis_failed"
    session_id: UUID
    original_prompt: str
    analysis: PromptAnalysis
    reason_codes: list[str] = Field(default_factory=list, description="Structured machine-readable reason codes.")
    processing_mode: str = Field(default="normal", description="Execution mode: 'normal' or 'long_input'.")
    mode: str = Field(default="professional", description="Enhancement mode requested: 'quick', 'professional', or 'deep'.")
    diagnostic_codes: list[str] = Field(default_factory=list, description="Diagnostic observation codes.")
    stage_timings: list[StageTimingDiagnostic] = Field(default_factory=list, description="Per-stage latency and token telemetry.")
    total_ms: Optional[float] = Field(default=None, description="Total measured end-to-end latency in milliseconds.")
    message: str = "HumanIntent could not verify that the analysis was grounded in your request."
    model: str
    history_saved: bool = True


EnhanceResponse = Annotated[
    Union[
        EnhancedResponse,
        ClarificationRequiredResponse,
        ValidationFailedResponse,
        AnalysisFailedResponse,
    ],
    Field(discriminator="status"),
]


class OllamaStatus(BaseModel):
    reachable: bool
    model: str


class HealthResponse(BaseModel):
    status: str
    ollama: OllamaStatus


class ModelsResponse(BaseModel):
    models: list[str]
    current_model: str
