from datetime import datetime
from typing import Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


FeedbackRatingType = Literal["HELPFUL", "NOT_HELPFUL"]


class FeedbackRequest(BaseModel):
    """Request payload to submit or update feedback for an enhancement session."""

    session_id: UUID = Field(
        ...,
        description="The UUID identifier of the enhancement session.",
    )
    rating: FeedbackRatingType = Field(
        ...,
        description="User rating: HELPFUL or NOT_HELPFUL.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional written feedback or improvement reason.",
    )
    edited_prompt: Optional[str] = Field(
        default=None,
        description="Optional user-edited improved prompt. Stored as feedback, NOT canonical truth.",
    )
    dataset_consent: bool = Field(
        default=False,
        description="Explicit consent for future model evaluation or dataset creation. Defaults strictly to False.",
    )


class FeedbackResponse(BaseModel):
    """Stored feedback entity returned to client."""

    id: str
    session_id: str
    rating: str
    reason: Optional[str] = None
    edited_prompt: Optional[str] = None
    dataset_consent: bool = False
    created_at: str
    updated_at: str
