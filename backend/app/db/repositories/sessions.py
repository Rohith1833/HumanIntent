import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple
from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session
from app.db.models import (
    AnalysisSnapshot,
    AttemptType,
    ClarificationRound,
    EnhancementAttempt,
    EnhancementSession,
    QualitySnapshot,
    SessionStatus,
    ValidationSnapshot,
)
from app.schemas.analysis import ClarificationAnswer, ClarificationQuestion, PromptAnalysis
from app.schemas.quality import PromptQualityReport
from app.schemas.validation import PromptValidation, ValidationSeverity


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_pydantic(obj: Any) -> Any:
    """Safely serialize Pydantic model or list of models into JSON-compatible dicts."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_serialize_pydantic(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_pydantic(v) for k, v in obj.items()}
    return obj


class EnhancementSessionRepository:
    """Repository managing enhancement sessions, authoritative inputs, and derived artifact snapshots."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_session(self, session_id: str) -> Optional[EnhancementSession]:
        """Retrieve a session by its UUID identifier."""
        stmt = select(EnhancementSession).where(EnhancementSession.id == session_id)
        return self.db.scalars(stmt).first()

    def get_or_create_session(
        self,
        session_id: Optional[str],
        original_prompt: str,
        model: Optional[str] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> Tuple[EnhancementSession, bool]:
        """
        Retrieve an existing session or create a new one.
        Ensures the authoritative original_prompt is NEVER overwritten.
        Returns (session, is_new).
        """
        if session_id:
            existing = self.get_session(session_id)
            if existing is not None:
                # Existing session: original_prompt is authoritative and preserved!
                # Only reuse if the incoming prompt matches the session's prompt to avoid contamination.
                if existing.original_prompt.strip().lower() == original_prompt.strip().lower():
                    existing.updated_at = _utc_now()
                    if model:
                        existing.model = model
                    existing.allow_reasonable_assumptions = allow_reasonable_assumptions
                    return existing, False
                else:
                    # Prompt mismatch: allocate a fresh isolated session for the new request
                    session_id = None

        # Create new session
        new_id = session_id if session_id else str(uuid.uuid4())
        session = EnhancementSession(
            id=new_id,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            original_prompt=original_prompt,
            status=SessionStatus.NEEDS_CLARIFICATION.value,
            mode="standard",
            model=model,
            allow_reasonable_assumptions=allow_reasonable_assumptions,
            history_saved=True,
            dataset_consent=False,
        )
        self.db.add(session)
        self.db.flush()
        return session, True

    def record_clarification_round(
        self,
        session: EnhancementSession,
        questions: Optional[List[ClarificationQuestion]] = None,
        answers: Optional[List[ClarificationAnswer]] = None,
        allow_reasonable_assumptions: bool = False,
    ) -> ClarificationRound:
        """Record structured clarification questions and/or authoritative human answers."""
        round_number = len(session.clarification_rounds) + 1 if session.clarification_rounds else 1

        round_record = ClarificationRound(
            id=str(uuid.uuid4()),
            session_id=session.id,
            round_number=round_number,
            created_at=_utc_now(),
            questions_json=_serialize_pydantic(questions),
            answers_json=_serialize_pydantic(answers),
            allow_reasonable_assumptions=allow_reasonable_assumptions,
        )
        self.db.add(round_record)
        self.db.flush()
        return round_record

    def record_analysis_snapshot(
        self,
        session: EnhancementSession,
        analysis: PromptAnalysis,
        round_number: Optional[int] = None,
    ) -> AnalysisSnapshot:
        """Record structured intent analysis snapshot per round."""
        if round_number is None:
            round_number = len(session.analysis_snapshots) + 1 if session.analysis_snapshots else 1

        grounding_data = _serialize_pydantic(getattr(analysis, "grounding", None))
        ambiguity_data = _serialize_pydantic(getattr(analysis, "ambiguity", None))
        is_grounded = True
        if hasattr(analysis, "grounding") and analysis.grounding is not None:
            is_grounded = bool(analysis.grounding.is_grounded)

        snapshot = AnalysisSnapshot(
            id=str(uuid.uuid4()),
            session_id=session.id,
            round_number=round_number,
            created_at=_utc_now(),
            analysis_json=_serialize_pydantic(analysis),
            grounding_json=grounding_data,
            ambiguity_json=ambiguity_data,
            classification=analysis.classification.category if analysis.classification else None,
            is_grounded=is_grounded,
        )
        self.db.add(snapshot)
        self.db.flush()
        return snapshot

    def record_enhancement_attempt(
        self,
        session: EnhancementSession,
        attempt_number: int,
        attempt_type: str,
        enhanced_prompt: str,
        template: Optional[str] = None,
        validation: Optional[PromptValidation] = None,
        quality: Optional[PromptQualityReport] = None,
    ) -> EnhancementAttempt:
        """Record an enhancement generation attempt, along with its validation and quality reports."""
        attempt = EnhancementAttempt(
            id=str(uuid.uuid4()),
            session_id=session.id,
            attempt_number=attempt_number,
            attempt_type=attempt_type,
            enhanced_prompt=enhanced_prompt,
            template=template,
            created_at=_utc_now(),
        )
        self.db.add(attempt)
        self.db.flush()

        if validation is not None:
            err_count = sum(1 for i in validation.issues if i.severity == ValidationSeverity.ERROR)
            warn_count = sum(1 for i in validation.issues if i.severity == ValidationSeverity.WARNING)
            val_snapshot = ValidationSnapshot(
                id=str(uuid.uuid4()),
                enhancement_attempt_id=attempt.id,
                validation_json=_serialize_pydantic(validation),
                is_valid=validation.is_valid,
                error_count=err_count,
                warning_count=warn_count,
                created_at=_utc_now(),
            )
            self.db.add(val_snapshot)

        # Quality snapshot is recorded ONLY when validation succeeds
        if quality is not None and validation is not None and validation.is_valid:
            ratio = getattr(quality, "inflation_ratio", None)
            risk = getattr(quality, "inflation_risk", None)
            risk_str = risk.value if hasattr(risk, "value") else str(risk) if risk else None
            q_snapshot = QualitySnapshot(
                id=str(uuid.uuid4()),
                enhancement_attempt_id=attempt.id,
                quality_json=_serialize_pydantic(quality),
                overall_score=float(quality.overall_score),
                inflation_ratio=ratio,
                inflation_risk=risk_str,
                created_at=_utc_now(),
            )
            self.db.add(q_snapshot)

        self.db.flush()
        return attempt

    def update_session_outcome(
        self,
        session: EnhancementSession,
        status: str,
        final_category: Optional[str] = None,
        final_template: Optional[str] = None,
        final_enhanced_prompt: Optional[str] = None,
        final_quality_score: Optional[float] = None,
        history_saved: bool = True,
    ) -> EnhancementSession:
        """Update session-level summary outcome fields."""
        session.status = status
        if final_category:
            session.final_category = final_category
        if final_template:
            session.final_template = final_template
        if final_enhanced_prompt is not None:
            session.final_enhanced_prompt = final_enhanced_prompt
        if final_quality_score is not None:
            session.final_quality_score = final_quality_score
        session.history_saved = history_saved
        session.updated_at = _utc_now()
        self.db.flush()
        return session

    def list_history_summaries(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[dict], int]:
        """
        List lightweight history summaries ordered by newest first.
        Deterministic prompt preview (first 100 characters). Zero LLM calls.
        Returns (list_of_summaries, total_count).
        """
        # Count total
        count_stmt = select(EnhancementSession.id)
        total = len(self.db.scalars(count_stmt).all())

        # Fetch page
        stmt = (
            select(EnhancementSession)
            .order_by(desc(EnhancementSession.created_at))
            .limit(limit)
            .offset(offset)
        )
        sessions = self.db.scalars(stmt).all()

        summaries = []
        for s in sessions:
            preview = s.original_prompt.strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."

            summaries.append(
                {
                    "session_id": s.id,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "original_prompt_preview": preview,
                    "status": s.status,
                    "category": s.final_category,
                    "quality_score": s.final_quality_score,
                }
            )

        return summaries, total

    def delete_session(self, session_id: str) -> bool:
        """Delete an enhancement session with all associated child snapshots."""
        session = self.get_session(session_id)
        if session is None:
            return False
        self.db.delete(session)
        self.db.commit()
        return True

    def clear_all(self) -> int:
        """Clear all local history sessions and associated snapshots."""
        count_stmt = select(EnhancementSession.id)
        ids = self.db.scalars(count_stmt).all()
        count = len(ids)

        self.db.execute(delete(EnhancementSession))
        self.db.commit()
        return count
