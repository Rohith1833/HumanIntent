import hashlib
import logging
from typing import List, Optional
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload

from dataset_factory.sanitizer import sanitize_text
from dataset_factory.schemas import (
    CurationStatus,
    DatasetRecord,
    NormalizedIntent,
    PrivacyMetadata,
    ProvenanceMetadata,
    QualityMetadata,
    RecordLabels,
    RejectionReason,
    SourceType,
)

logger = logging.getLogger("dataset_factory.real_loader")


def _hash_session_id(session_id: str) -> str:
    """Generate a one-way pseudonymous identifier for exported records."""
    salt = "humanintent_provenance_v1"
    digest = hashlib.sha256(f"{session_id}:{salt}".encode("utf-8")).hexdigest()[:16]
    return f"real_{digest}"


def load_consented_real_data(
    database_url: str,
    dataset_version: str = "humanintent-v0.1.0",
) -> List[DatasetRecord]:
    """Load exclusively consented sessions from Phase 8 SQLite history.

    CRITICAL PRIVACY RULE: Only sessions where dataset_consent == True
    (either at session level or explicit feedback level) are queried.
    Unconsented sessions are filtered at the SQL database query level.
    """
    # Import db models from backend package
    from app.db.models import EnhancementSession, Feedback

    engine = create_engine(database_url, echo=False)
    records: List[DatasetRecord] = []

    with Session(engine) as db:
        # Strict SQL-level filtering for dataset_consent == True
        stmt = (
            select(EnhancementSession)
            .outerjoin(Feedback, EnhancementSession.id == Feedback.session_id)
            .where(
                (EnhancementSession.dataset_consent == True)  # noqa: E712
                | (Feedback.dataset_consent == True)          # noqa: E712
            )
            .options(
                selectinload(EnhancementSession.feedback),
                selectinload(EnhancementSession.analysis_snapshots),
                selectinload(EnhancementSession.enhancement_attempts),
            )
        )

        consented_sessions = db.scalars(stmt).unique().all()
        logger.info("Found %d consented real session(s) in database.", len(consented_sessions))

        for session in consented_sessions:
            # Re-verify consent explicitly in memory
            has_explicit_consent = bool(
                session.dataset_consent
                or (session.feedback and session.feedback.dataset_consent)
            )
            if not has_explicit_consent:
                continue

            raw_prompt = session.original_prompt
            if not raw_prompt or not raw_prompt.strip():
                continue

            # Apply privacy sanitizer
            sanitized_res = sanitize_text(raw_prompt)
            if sanitized_res.is_secret_risk:
                logger.warning(
                    "Excluding consented session %s due to unredactable secret: %s",
                    session.id,
                    sanitized_res.risk_reason,
                )
                continue

            # Pseudonymized source identifier (no raw UUID export)
            pseudo_source_id = _hash_session_id(session.id)

            # Determine enhanced prompt: user edit if available and preferred, else system enhanced
            chosen_prompt = None
            labels = RecordLabels()

            if session.feedback and session.feedback.edited_prompt:
                edited_sanitized = sanitize_text(session.feedback.edited_prompt)
                if not edited_sanitized.is_secret_risk:
                    chosen_prompt = edited_sanitized.text
                    labels.self_correction = True

            if not chosen_prompt and session.final_enhanced_prompt:
                enhanced_sanitized = sanitize_text(session.final_enhanced_prompt)
                if not enhanced_sanitized.is_secret_risk:
                    chosen_prompt = enhanced_sanitized.text

            # Check if clarification occurred
            clarification_required = session.status == "needs_clarification"
            if clarification_required:
                labels.ambiguous = True

            # Extract latest analysis if available
            normalized_intent = None
            if session.analysis_snapshots:
                latest_snap = session.analysis_snapshots[-1]
                analysis_data = latest_snap.analysis_json or {}
                normalized_intent = NormalizedIntent(
                    objective=analysis_data.get("objective", ""),
                    explicit_requirements=analysis_data.get("explicit_requirements", []),
                    constraints=analysis_data.get("constraints", []),
                    preferences=analysis_data.get("preferences", []),
                    missing_context=analysis_data.get("missing_context", []),
                )

            quality_score = session.final_quality_score or 0.0
            quality_meta = QualityMetadata(
                validation_passed=(session.status == "enhanced"),
                quality_score=quality_score,
            )

            record = DatasetRecord(
                dataset_version=dataset_version,
                source_type=SourceType.CONSENTED_REAL,
                source_id=pseudo_source_id,
                category=session.final_category or "GENERAL",
                original_prompt=sanitized_res.text,
                normalized_intent=normalized_intent,
                enhanced_prompt=chosen_prompt,
                clarification_required=clarification_required,
                quality=quality_meta,
                provenance=ProvenanceMetadata(
                    source_id=pseudo_source_id,
                    teacher_pipeline_version="humanintent-v1.0",
                    dataset_version=dataset_version,
                ),
                privacy=PrivacyMetadata(
                    sanitized=sanitized_res.sanitized,
                    pii_types_removed=sanitized_res.pii_types_removed,
                ),
                labels=labels,
                curation_status=CurationStatus.ELIGIBLE,
            )
            records.append(record)

    return records
