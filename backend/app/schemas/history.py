from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.analysis import ClarificationAnswer, ClarificationQuestion, PromptAnalysis
from app.schemas.feedback import FeedbackResponse
from app.schemas.quality import PromptQualityReport
from app.schemas.validation import PromptValidation


class HistorySummaryItem(BaseModel):
    """Lightweight session summary for history list views."""

    session_id: str
    created_at: str
    updated_at: str
    original_prompt_preview: str = Field(
        ...,
        description="Deterministic preview of the original prompt (first 100 characters). Zero LLM calls.",
    )
    status: str
    category: Optional[str] = None
    quality_score: Optional[float] = None


class HistoryListResponse(BaseModel):
    """Paginated list of historical enhancement sessions."""

    items: List[HistorySummaryItem]
    total: int
    limit: int
    offset: int


class ClarificationRoundDetail(BaseModel):
    """Details of a clarification round."""

    round_number: int
    created_at: str
    questions: Optional[List[ClarificationQuestion]] = None
    answers: Optional[List[ClarificationAnswer]] = None
    allow_reasonable_assumptions: bool = False


class EnhancementAttemptDetail(BaseModel):
    """Audit detail of an enhancement attempt."""

    attempt_number: int
    attempt_type: str
    enhanced_prompt: str
    template: Optional[str] = None
    validation: Optional[PromptValidation] = None
    quality: Optional[PromptQualityReport] = None


class HistoryDetailResponse(BaseModel):
    """Reconstructed user-visible result from local history. Zero LLM calls."""

    session_id: str
    created_at: str
    updated_at: str
    original_prompt: str
    status: str
    mode: Optional[str] = None
    model: Optional[str] = None
    allow_reasonable_assumptions: bool = False
    category: Optional[str] = None
    template: Optional[str] = None
    clarification_rounds: List[ClarificationRoundDetail] = []
    latest_analysis: Optional[PromptAnalysis] = None
    final_enhanced_prompt: Optional[str] = None
    validation: Optional[PromptValidation] = None
    quality: Optional[PromptQualityReport] = None
    attempts: List[EnhancementAttemptDetail] = []
    feedback: Optional[FeedbackResponse] = None
    history_saved: bool = True
    dataset_consent: bool = False


class DeleteResponse(BaseModel):
    """Response returned upon history deletion."""

    status: str
    session_id: Optional[str] = None
    deleted_count: Optional[int] = None
