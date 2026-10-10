import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db.models import EnhancementSession, Feedback


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FeedbackRepository:
    """Repository managing user feedback and privacy/consent metadata per session."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_feedback(self, session_id: str) -> Optional[Feedback]:
        """Retrieve feedback record for a given session."""
        stmt = select(Feedback).where(Feedback.session_id == session_id)
        return self.db.scalars(stmt).first()

    def upsert_feedback(
        self,
        session_id: str,
        rating: str,
        reason: Optional[str] = None,
        edited_prompt: Optional[str] = None,
        dataset_consent: bool = False,
    ) -> Feedback:
        """
        Upsert feedback for an enhancement session.
        Ensures:
        - Rating is required (HELPFUL / NOT_HELPFUL)
        - Reason and edited_prompt are optional
        - edited_prompt is stored as feedback, NOT overwriting canonical prompt
        - dataset_consent defaults to False and is explicitly tracked
        - Updates session-level dataset_consent flag
        """
        # Ensure session exists
        session_stmt = select(EnhancementSession).where(EnhancementSession.id == session_id)
        session = self.db.scalars(session_stmt).first()
        if session is None:
            raise ValueError(f"EnhancementSession with id '{session_id}' not found.")

        existing = self.get_feedback(session_id)
        if existing is not None:
            existing.rating = rating
            existing.reason = reason
            existing.edited_prompt = edited_prompt
            existing.dataset_consent = dataset_consent
            existing.updated_at = _utc_now()
            # Sync session-level consent flag
            session.dataset_consent = dataset_consent
            self.db.commit()
            self.db.refresh(existing)
            return existing

        new_feedback = Feedback(
            id=str(uuid.uuid4()),
            session_id=session_id,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            rating=rating,
            reason=reason,
            edited_prompt=edited_prompt,
            dataset_consent=dataset_consent,
        )
        self.db.add(new_feedback)
        # Sync session-level consent flag
        session.dataset_consent = dataset_consent
        self.db.commit()
        self.db.refresh(new_feedback)
        return new_feedback
