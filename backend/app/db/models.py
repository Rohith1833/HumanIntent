import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class SessionStatus(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    ENHANCED = "enhanced"
    ANALYSIS_FAILED = "analysis_failed"
    VALIDATION_FAILED = "validation_failed"


class FeedbackRating(str, Enum):
    HELPFUL = "HELPFUL"
    NOT_HELPFUL = "NOT_HELPFUL"


class AttemptType(str, Enum):
    INITIAL = "INITIAL"
    REPAIR = "REPAIR"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EnhancementSession(Base):
    """Persistent session record representing a user's prompt enhancement journey."""

    __tablename__ = "enhancement_sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        index=True,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        onupdate=_utc_now,
        index=True,
        nullable=False,
    )

    # AUTHORITATIVE: Original human prompt, preserved exactly, never overwritten.
    original_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    mode: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        default="standard",
    )
    model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    allow_reasonable_assumptions: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Derived summary fields for fast history queries
    final_category: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    final_template: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    final_enhanced_prompt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    final_quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # Persistence & Privacy Flags
    history_saved: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    dataset_consent: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Relationships with cascading deletion
    clarification_rounds: Mapped[List["ClarificationRound"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ClarificationRound.round_number",
    )
    analysis_snapshots: Mapped[List["AnalysisSnapshot"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AnalysisSnapshot.round_number",
    )
    enhancement_attempts: Mapped[List["EnhancementAttempt"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="EnhancementAttempt.attempt_number",
    )
    feedback: Mapped[Optional["Feedback"]] = relationship(
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_enhancement_sessions_created_desc", created_at.desc()),
    )


class ClarificationRound(Base):
    """Audit record for clarification questions and authoritative human answers."""

    __tablename__ = "clarification_rounds"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("enhancement_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    round_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        nullable=False,
    )

    # Structured question & answer data
    questions_json: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
    )
    # AUTHORITATIVE: User's clarification answers
    answers_json: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
    )
    allow_reasonable_assumptions: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    session: Mapped["EnhancementSession"] = relationship(
        back_populates="clarification_rounds",
    )


class AnalysisSnapshot(Base):
    """Audit snapshot of intent analysis per round."""

    __tablename__ = "analysis_snapshots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("enhancement_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    round_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        nullable=False,
    )

    analysis_json: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
    )
    grounding_json: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
    )
    ambiguity_json: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
    )
    classification: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    is_grounded: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    session: Mapped["EnhancementSession"] = relationship(
        back_populates="analysis_snapshots",
    )


class EnhancementAttempt(Base):
    """Audit record of generated enhancement attempts (initial and repair)."""

    __tablename__ = "enhancement_attempts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("enhancement_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    attempt_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=AttemptType.INITIAL.value,
    )
    enhanced_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    template: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        nullable=False,
    )

    session: Mapped["EnhancementSession"] = relationship(
        back_populates="enhancement_attempts",
    )
    validation: Mapped[Optional["ValidationSnapshot"]] = relationship(
        back_populates="attempt",
        uselist=False,
        cascade="all, delete-orphan",
    )
    quality: Mapped[Optional["QualitySnapshot"]] = relationship(
        back_populates="attempt",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ValidationSnapshot(Base):
    """Validation result attached to a specific enhancement attempt."""

    __tablename__ = "validation_snapshots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    enhancement_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("enhancement_attempts.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    validation_json: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
    )
    is_valid: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    error_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    warning_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        nullable=False,
    )

    attempt: Mapped["EnhancementAttempt"] = relationship(
        back_populates="validation",
    )


class QualitySnapshot(Base):
    """Quality report attached only to a successful valid enhancement attempt."""

    __tablename__ = "quality_snapshots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    enhancement_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("enhancement_attempts.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    quality_json: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
    )
    overall_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    inflation_ratio: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    inflation_risk: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        nullable=False,
    )

    attempt: Mapped["EnhancementAttempt"] = relationship(
        back_populates="quality",
    )


class Feedback(Base):
    """User feedback entity for an enhancement session."""

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("enhancement_sessions.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        onupdate=_utc_now,
        nullable=False,
    )

    rating: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # DERIVED / FEEDBACK: Optional user-edited improved prompt. Not canonical truth!
    edited_prompt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # PRIVACY PRINCIPLE: Default False. Explicit consent required for future training/export.
    dataset_consent: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    session: Mapped["EnhancementSession"] = relationship(
        back_populates="feedback",
    )
