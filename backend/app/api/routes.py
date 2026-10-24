import logging
import time
import uuid
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.core.analyzer import prompt_analyzer
from app.core.config import settings
from app.core.context_budget import (
    LONG_INPUT_MODE,
    NORMAL_CONTEXT_MODE,
    ProcessingMode,
    context_budget_manager,
)
from app.core.enhancer import prompt_enhancer
from app.core.fast_path import fast_path_pipeline
from app.db.repositories.feedback import FeedbackRepository
from app.db.repositories.sessions import EnhancementSessionRepository
from app.db.session import get_db
from app.llm.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    ollama_client,
)
from app.schemas.analysis import AnalyzeRequest, ClarificationAnswer, ClarificationQuestion, PromptAnalysis
from app.schemas.enhance import (
    AnalysisFailedResponse,
    ClarificationRequiredResponse,
    EnhancedResponse,
    EnhanceRequest,
    EnhanceResponse,
    HealthResponse,
    ModelsResponse,
    OllamaStatus,
    ValidationFailedResponse,
)
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.schemas.history import (
    ClarificationRoundDetail,
    DeleteResponse,
    EnhancementAttemptDetail,
    HistoryDetailResponse,
    HistoryListResponse,
    HistorySummaryItem,
)
from app.schemas.quality import PromptQualityReport
from app.schemas.validation import PromptValidation, ValidationSeverity
from app.templates.selector import get_template

logger = logging.getLogger("humanintent.api")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return backend status and Ollama connectivity status."""
    is_reachable = await ollama_client.check_health()
    return HealthResponse(
        status="ok",
        ollama=OllamaStatus(
            reachable=is_reachable,
            model=settings.ollama_model,
        ),
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List locally installed models from Ollama."""
    try:
        models = await ollama_client.list_models()
        return ModelsResponse(
            models=models,
            current_model=settings.ollama_model,
        )
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


def _clean_ollama_error_detail(exc: OllamaResponseError) -> str:
    """Ensure user-facing error messages are clean and never leak raw Pydantic validation tracebacks."""
    err_str = str(exc)
    if any(keyword in err_str.lower() for keyword in ["validation error", "required schema", "invalid json", "structured response"]):
        return "HumanIntent couldn't produce a valid structured analysis for this request. Please try again or rephrase the request."
    return err_str


@router.post("/analyze", response_model=PromptAnalysis)
async def analyze_prompt(payload: AnalyzeRequest) -> PromptAnalysis:
    """Standalone endpoint to extract structured objective, requirements, and missing context."""
    try:
        return await prompt_analyzer.analyze(payload.prompt)
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OllamaModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_clean_ollama_error_detail(exc),
        ) from exc


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance_prompt(
    payload: EnhanceRequest,
    db: Session = Depends(get_db),
) -> EnhanceResponse:
    """
    Enhance the user prompt by analyzing intent, clarifying if critically ambiguous, or producing an enhanced prompt.
    Persists session and audit snapshots in a short SQLite transaction when history is enabled.
    """
    session_id_str = str(payload.session_id) if payload.session_id else None
    session = None
    session_repo = EnhancementSessionRepository(db)

    # 1. Prepare session record if history is enabled
    is_new_session = True
    if settings.history_enabled:
        try:
            session, is_new_session = session_repo.get_or_create_session(
                session_id=session_id_str,
                original_prompt=payload.prompt,
                model=settings.ollama_model,
                allow_reasonable_assumptions=payload.allow_reasonable_assumptions,
            )
            active_session_id = session.id
            if session_id_str and is_new_session:
                logger.warning(
                    "Request state isolation: requested session %s prompt mismatched. Allocated fresh isolated session %s",
                    session_id_str,
                    active_session_id,
                )
        except Exception as exc:
            logger.error("Failed to initialize session record in database: %s", exc)
            active_session_id = session_id_str or str(uuid.uuid4())
    else:
        active_session_id = session_id_str or str(uuid.uuid4())

    # Only supply clarification answers if they belong to this active session continuation
    effective_clarification_answers = payload.clarification_answers if not (session_id_str and is_new_session) else None

    # 2. Run AI enhancement pipeline (Call budgets strictly preserved, NO locks held during inference)
    t_start = time.perf_counter()
    try:
        if payload.mode is not None:
            result = await fast_path_pipeline.enhance_fast_path(
                user_prompt=payload.prompt,
                clarification_answers=effective_clarification_answers,
                allow_reasonable_assumptions=payload.allow_reasonable_assumptions,
                mode=payload.mode,
            )
        else:
            result = await prompt_enhancer.enhance(
                user_prompt=payload.prompt,
                clarification_answers=effective_clarification_answers,
                allow_reasonable_assumptions=payload.allow_reasonable_assumptions,
            )
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OllamaModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_clean_ollama_error_detail(exc),
        ) from exc

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    measured_total_ms = getattr(result, "total_ms", None) or elapsed_ms
    measured_inference_ms = getattr(result, "inference_ms", None)

    analysis, enhanced_text = result[0], result[1]
    template_obj = get_template(analysis.classification.category) if analysis.classification else None
    template_name = template_obj.name if template_obj else None
    validation = getattr(result, "validation", None)
    quality = getattr(result, "quality", None)
    repair_attempted = getattr(result, "repair_attempted", False)
    initial_prompt = getattr(result, "initial_enhanced_prompt", enhanced_text)
    initial_val = getattr(result, "initial_validation", validation)
    stage_timings = getattr(result, "stage_timings", [])

    # 3. Determine pipeline outcome status, processing mode, and diagnostic codes
    stage_report = context_budget_manager.check_stage_budget("analyzer", "", payload.prompt)
    is_long_mode = (
        (hasattr(analysis, "grounding") and analysis.grounding and len(getattr(analysis.grounding, "source_units", [])) > 6 and len(payload.prompt) > 2000)
        or stage_report.processing_mode == ProcessingMode.LONG_INPUT
    )
    active_mode = "long_input" if is_long_mode else "normal"
    diagnostic_codes = [LONG_INPUT_MODE] if is_long_mode else [NORMAL_CONTEXT_MODE]
    response_mode = getattr(result, "mode", None) or payload.mode or "professional"

    outcome_status: str
    response_obj: EnhanceResponse

    if hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded:
        outcome_status = "analysis_failed"
        reason_codes = ["ANALYSIS_UNGROUNDED"]
        if getattr(analysis.grounding, "missing_source_units", None):
            reason_codes.extend(["SOURCE_REQUIREMENT_DROPPED", "ANALYSIS_INCOMPLETE", "SOURCE_COVERAGE_FAILED"])
        if session_id_str and is_new_session:
            reason_codes.append("REQUEST_STATE_CONTAMINATION")
        logger.warning("Pipeline outcome 'analysis_failed' with reason codes: %s", reason_codes)
        response_obj = AnalysisFailedResponse(
            status="analysis_failed",
            session_id=UUID(active_session_id),
            original_prompt=payload.prompt,
            analysis=analysis,
            reason_codes=reason_codes,
            processing_mode=active_mode,
            mode=response_mode,
            diagnostic_codes=diagnostic_codes,
            stage_timings=stage_timings,
            total_ms=measured_total_ms,
            message="HumanIntent could not verify that the analysis was grounded in your request.",
            model=settings.ollama_model,
            history_saved=False,  # updated after commit
        )
    elif analysis.ambiguity.needs_clarification and not payload.allow_reasonable_assumptions:
        outcome_status = "needs_clarification"
        reason_codes = ["CRITICAL_AMBIGUITY"]
        if session_id_str and is_new_session:
            reason_codes.append("REQUEST_STATE_CONTAMINATION")
        logger.info("Pipeline outcome 'needs_clarification' with reason codes: %s", reason_codes)
        response_obj = ClarificationRequiredResponse(
            status="needs_clarification",
            session_id=UUID(active_session_id),
            original_prompt=payload.prompt,
            analysis=analysis,
            questions=analysis.ambiguity.questions,
            reason_codes=reason_codes,
            processing_mode=active_mode,
            mode=response_mode,
            diagnostic_codes=diagnostic_codes,
            stage_timings=stage_timings,
            total_ms=measured_total_ms,
            model=settings.ollama_model,
            history_saved=False,
        )
    elif validation is not None and not validation.is_valid:
        outcome_status = "validation_failed"
        reason_codes = []
        if repair_attempted:
            reason_codes.append("VALIDATION_REPAIR_FAILED")
        if session_id_str and is_new_session:
            reason_codes.append("REQUEST_STATE_CONTAMINATION")
        for issue in validation.issues:
            if issue.severity == ValidationSeverity.ERROR:
                code_name = issue.type.value.upper()
                if code_name not in reason_codes:
                    reason_codes.append(code_name)
        logger.warning("Pipeline outcome 'validation_failed' with reason codes: %s", reason_codes)
        response_obj = ValidationFailedResponse(
            status="validation_failed",
            session_id=UUID(active_session_id),
            original_prompt=payload.prompt,
            analysis=analysis,
            validation=validation,
            reason_codes=reason_codes,
            processing_mode=active_mode,
            mode=response_mode,
            diagnostic_codes=diagnostic_codes,
            stage_timings=stage_timings,
            total_ms=measured_total_ms,
            message="HumanIntent could not produce a sufficiently faithful enhanced prompt after one repair attempt.",
            model=settings.ollama_model,
            history_saved=False,
        )
    else:
        outcome_status = "enhanced"
        response_obj = EnhancedResponse(
            status="enhanced",
            session_id=UUID(active_session_id),
            original_prompt=payload.prompt,
            analysis=analysis,
            enhanced_prompt=enhanced_text,
            model=settings.ollama_model,
            template=template_name,
            validation=validation,
            quality=quality,
            processing_mode=active_mode,
            mode=response_mode,
            diagnostic_codes=diagnostic_codes,
            stage_timings=stage_timings,
            total_ms=measured_total_ms,
            inference_ms=measured_inference_ms,
            history_saved=False,
        )

    # 4. Short DB transaction for persistence (after inference completes)
    history_saved = False
    if settings.history_enabled and session is not None:
        try:
            # Record user clarification answers if provided in this round
            if payload.clarification_answers:
                session_repo.record_clarification_round(
                    session=session,
                    answers=payload.clarification_answers,
                    allow_reasonable_assumptions=payload.allow_reasonable_assumptions,
                )

            # Record analysis snapshot
            session_repo.record_analysis_snapshot(
                session=session,
                analysis=analysis,
            )

            if outcome_status == "analysis_failed":
                session_repo.update_session_outcome(
                    session=session,
                    status=outcome_status,
                    final_category=analysis.classification.category if analysis.classification else None,
                    history_saved=True,
                )
            elif outcome_status == "needs_clarification":
                # Record asked clarification questions in the round
                session_repo.record_clarification_round(
                    session=session,
                    questions=analysis.ambiguity.questions if analysis.ambiguity else None,
                    allow_reasonable_assumptions=payload.allow_reasonable_assumptions,
                )
                session_repo.update_session_outcome(
                    session=session,
                    status=outcome_status,
                    final_category=analysis.classification.category if analysis.classification else None,
                    history_saved=True,
                )
            elif outcome_status == "validation_failed":
                session_repo.record_enhancement_attempt(
                    session=session,
                    attempt_number=1,
                    attempt_type="INITIAL",
                    enhanced_prompt=initial_prompt or "",
                    template=template_name,
                    validation=initial_val,
                    quality=None,
                )
                if repair_attempted:
                    session_repo.record_enhancement_attempt(
                        session=session,
                        attempt_number=2,
                        attempt_type="REPAIR",
                        enhanced_prompt=enhanced_text,
                        template=template_name,
                        validation=validation,
                        quality=None,
                    )
                session_repo.update_session_outcome(
                    session=session,
                    status=outcome_status,
                    final_category=analysis.classification.category if analysis.classification else None,
                    final_template=template_name,
                    final_enhanced_prompt=enhanced_text,
                    history_saved=True,
                )
            elif outcome_status == "enhanced":
                overall_score = quality.overall_score if quality else None
                if repair_attempted:
                    # Attempt 1 (INITIAL, failed validation)
                    session_repo.record_enhancement_attempt(
                        session=session,
                        attempt_number=1,
                        attempt_type="INITIAL",
                        enhanced_prompt=initial_prompt or "",
                        template=template_name,
                        validation=initial_val,
                        quality=None,
                    )
                    # Attempt 2 (REPAIR, passed validation + quality)
                    session_repo.record_enhancement_attempt(
                        session=session,
                        attempt_number=2,
                        attempt_type="REPAIR",
                        enhanced_prompt=enhanced_text,
                        template=template_name,
                        validation=validation,
                        quality=quality,
                    )
                else:
                    session_repo.record_enhancement_attempt(
                        session=session,
                        attempt_number=1,
                        attempt_type="INITIAL",
                        enhanced_prompt=enhanced_text,
                        template=template_name,
                        validation=validation,
                        quality=quality,
                    )
                session_repo.update_session_outcome(
                    session=session,
                    status=outcome_status,
                    final_category=analysis.classification.category if analysis.classification else None,
                    final_template=template_name,
                    final_enhanced_prompt=enhanced_text,
                    final_quality_score=overall_score,
                    history_saved=True,
                )

            db.commit()
            history_saved = True
            logger.info("Session %s persisted (status=%s, category=%s)", active_session_id, outcome_status, analysis.classification.category if analysis.classification else "unknown")
        except Exception as exc:
            db.rollback()
            logger.error("Failed to persist session %s: %s", active_session_id, exc)
            history_saved = False

    # Update response history_saved attribute
    response_obj.history_saved = history_saved
    return response_obj


@router.get("/history", response_model=HistoryListResponse)
def get_history(
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    """Retrieve paginated lightweight history summaries. Zero LLM calls."""
    clamped_limit = min(limit, 100)
    session_repo = EnhancementSessionRepository(db)
    summaries, total = session_repo.list_history_summaries(limit=clamped_limit, offset=offset)
    items = [HistorySummaryItem(**s) for s in summaries]
    return HistoryListResponse(
        items=items,
        total=total,
        limit=clamped_limit,
        offset=offset,
    )


@router.get("/history/{session_id}", response_model=HistoryDetailResponse)
def get_history_detail(
    session_id: str,
    db: Session = Depends(get_db),
) -> HistoryDetailResponse:
    """Reconstruct complete historical enhancement result. Zero LLM calls."""
    session_repo = EnhancementSessionRepository(db)
    session = session_repo.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enhancement session '{session_id}' not found.",
        )

    # Reconstruct clarification rounds
    clarification_rounds = []
    for r in session.clarification_rounds:
        q_objs = [ClarificationQuestion(**q) for q in r.questions_json] if r.questions_json else None
        a_objs = [ClarificationAnswer(**a) for a in r.answers_json] if r.answers_json else None
        clarification_rounds.append(
            ClarificationRoundDetail(
                round_number=r.round_number,
                created_at=r.created_at.isoformat(),
                questions=q_objs,
                answers=a_objs,
                allow_reasonable_assumptions=r.allow_reasonable_assumptions,
            )
        )

    # Latest analysis
    latest_analysis = None
    if session.analysis_snapshots:
        last_snapshot = session.analysis_snapshots[-1]
        try:
            latest_analysis = PromptAnalysis.model_validate(last_snapshot.analysis_json)
        except Exception:
            latest_analysis = None

    # Latest validation and quality from attempts
    latest_validation = None
    latest_quality = None
    attempts_detail = []
    for att in session.enhancement_attempts:
        v_obj = None
        q_obj = None
        if att.validation and att.validation.validation_json:
            try:
                v_obj = PromptValidation.model_validate(att.validation.validation_json)
                latest_validation = v_obj
            except Exception:
                pass
        if att.quality and att.quality.quality_json:
            try:
                q_obj = PromptQualityReport.model_validate(att.quality.quality_json)
                latest_quality = q_obj
            except Exception:
                pass
        attempts_detail.append(
            EnhancementAttemptDetail(
                attempt_number=att.attempt_number,
                attempt_type=att.attempt_type,
                enhanced_prompt=att.enhanced_prompt,
                template=att.template,
                validation=v_obj,
                quality=q_obj,
            )
        )

    # Feedback
    feedback_detail = None
    if session.feedback:
        fb = session.feedback
        feedback_detail = FeedbackResponse(
            id=fb.id,
            session_id=fb.session_id,
            rating=fb.rating,
            reason=fb.reason,
            edited_prompt=fb.edited_prompt,
            dataset_consent=fb.dataset_consent,
            created_at=fb.created_at.isoformat(),
            updated_at=fb.updated_at.isoformat(),
        )

    return HistoryDetailResponse(
        session_id=session.id,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        original_prompt=session.original_prompt,
        status=session.status,
        mode=session.mode,
        model=session.model,
        allow_reasonable_assumptions=session.allow_reasonable_assumptions,
        category=session.final_category,
        template=session.final_template,
        clarification_rounds=clarification_rounds,
        latest_analysis=latest_analysis,
        final_enhanced_prompt=session.final_enhanced_prompt,
        validation=latest_validation,
        quality=latest_quality,
        attempts=attempts_detail,
        feedback=feedback_detail,
        history_saved=session.history_saved,
        dataset_consent=session.dataset_consent,
    )


@router.delete("/history/{session_id}", response_model=DeleteResponse)
def delete_history_item(
    session_id: str,
    db: Session = Depends(get_db),
) -> DeleteResponse:
    """Delete a single session and cascade delete all associated audit snapshots."""
    session_repo = EnhancementSessionRepository(db)
    deleted = session_repo.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enhancement session '{session_id}' not found.",
        )
    return DeleteResponse(
        status="deleted",
        session_id=session_id,
    )


@router.delete("/history", response_model=DeleteResponse)
def clear_history(
    db: Session = Depends(get_db),
) -> DeleteResponse:
    """Clear all local history sessions and child snapshots."""
    session_repo = EnhancementSessionRepository(db)
    count = session_repo.clear_all()
    return DeleteResponse(
        status="cleared",
        deleted_count=count,
    )


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Submit or update feedback for an enhancement session. Upserts cleanly without duplicates."""
    feedback_repo = FeedbackRepository(db)
    try:
        fb = feedback_repo.upsert_feedback(
            session_id=str(payload.session_id),
            rating=payload.rating,
            reason=payload.reason,
            edited_prompt=payload.edited_prompt,
            dataset_consent=payload.dataset_consent,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return FeedbackResponse(
        id=fb.id,
        session_id=fb.session_id,
        rating=fb.rating,
        reason=fb.reason,
        edited_prompt=fb.edited_prompt,
        dataset_consent=fb.dataset_consent,
        created_at=fb.created_at.isoformat(),
        updated_at=fb.updated_at.isoformat(),
    )
