from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from app.schemas.analysis import ClarificationAnswer, ClarificationQuestion, PromptAnalysis


class EnhanceRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The rough human prompt to enhance.",
    )
    clarification_answers: Optional[list[ClarificationAnswer]] = Field(
        default=None,
        description="User answers to previously asked clarification questions.",
    )
    allow_reasonable_assumptions: bool = Field(
        default=False,
        description="If True, bypasses clarification and instructs the enhancer to proceed with safe, minimal assumptions.",
    )

    @field_validator("prompt")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return stripped


from app.schemas.quality import PromptQualityReport
from app.schemas.validation import PromptValidation


class EnhancedResponse(BaseModel):
    status: Literal["enhanced"] = "enhanced"
    original_prompt: str
    analysis: PromptAnalysis
    enhanced_prompt: str
    model: str
    template: Optional[str] = None
    validation: Optional[PromptValidation] = None
    quality: Optional[PromptQualityReport] = None


class ClarificationRequiredResponse(BaseModel):
    status: Literal["needs_clarification"] = "needs_clarification"
    original_prompt: str
    analysis: PromptAnalysis
    questions: list[ClarificationQuestion]
    model: str


class ValidationFailedResponse(BaseModel):
    status: Literal["validation_failed"] = "validation_failed"
    original_prompt: str
    analysis: PromptAnalysis
    validation: PromptValidation
    message: str = "HumanIntent could not produce a sufficiently faithful enhanced prompt after one repair attempt."
    model: str


class AnalysisFailedResponse(BaseModel):
    status: Literal["analysis_failed"] = "analysis_failed"
    original_prompt: str
    analysis: PromptAnalysis
    message: str = "HumanIntent could not verify that the analysis was grounded in your request."
    model: str


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
