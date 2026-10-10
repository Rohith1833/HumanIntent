import os
import tempfile
import uuid
from typing import Generator
from unittest.mock import AsyncMock, patch
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from app.core.config import settings
from app.db.base import Base
from app.db.models import (
    AnalysisSnapshot,
    ClarificationRound,
    EnhancementAttempt,
    EnhancementSession,
    Feedback,
    QualitySnapshot,
    SessionStatus,
    ValidationSnapshot,
)
from app.db.repositories.feedback import FeedbackRepository
from app.db.repositories.sessions import EnhancementSessionRepository
from app.db.session import get_db
from app.main import app
from app.schemas.analysis import (
    AmbiguityAssessment,
    AnalysisGrounding,
    ClarificationAnswer,
    ClarificationQuestion,
    PromptAnalysis,
    PromptClassification,
)
from app.schemas.quality import InflationRisk, PromptQualityReport, QualityDimension
from app.schemas.validation import (
    PromptValidation,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
)


@pytest.fixture
def test_db():
    """Create an isolated, in-memory SQLite database for testing repositories and API."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db):
    """FastAPI TestClient with isolated test database dependency injection."""
    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==============================================================================
# Helper Mock Objects
# ==============================================================================

def make_analysis(
    prompt: str = "test prompt",
    category: str = "CODING",
    needs_clarification: bool = False,
    is_grounded: bool = True,
    questions: list = None,
) -> PromptAnalysis:
    qs = questions or []
    if needs_clarification and not qs:
        qs = [
            ClarificationQuestion(
                id="q1",
                question="What language should be used?",
                reason="Language unspecified",
                target_dimension="tech_stack",
            )
        ]
    return PromptAnalysis(
        classification=PromptClassification(category=category, confidence=0.95),
        objective=f"Objective for {prompt}",
        explicit_requirements=["Req 1"],
        constraints=["Constraint 1"],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(
            score=70 if needs_clarification else 10,
            level="HIGH" if needs_clarification else "LOW",
            needs_clarification=needs_clarification,
            reasons=["Needs stack clarification"] if needs_clarification else [],
            questions=qs,
        ),
        grounding=AnalysisGrounding(
            is_grounded=is_grounded,
            evidence=[],
            unsupported_items=[] if is_grounded else ["unsupported detail"],
        ),
    )


def make_validation(is_valid: bool = True) -> PromptValidation:
    issues = []
    if not is_valid:
        issues.append(
            ValidationIssue(
                type=ValidationIssueType.INVENTED_REQUIREMENT,
                severity=ValidationSeverity.ERROR,
                message="Hallucinated unrequested requirement",
            )
        )
    return PromptValidation(
        is_valid=is_valid,
        validation_score=0.95 if is_valid else 0.40,
        issues=issues,
        intent_preserved=is_valid,
        requirements_preserved=is_valid,
    )


def make_quality(score: int = 92) -> PromptQualityReport:
    dim = QualityDimension(score=score, label="Excellent", reason="Good")
    return PromptQualityReport(
        overall_score=score,
        overall_label="Excellent",
        intent_preservation=dim,
        requirement_coverage=dim,
        constraint_preservation=dim,
        clarity=dim,
        specificity=dim,
        output_definition=dim,
        structure=dim,
        concision=dim,
        inflation_ratio=1.5,
        inflation_risk=InflationRisk.LOW,
    )


# ==============================================================================
# TEST GROUP A — MIGRATION
# ==============================================================================

def test_group_a_alembic_migration():
    """Verify alembic migrations run cleanly against a fresh, empty SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        tmp_db_path = tmp_file.name

    try:
        db_url = f"sqlite:///{tmp_db_path}"
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # Run migration upgrade to head
        command.upgrade(alembic_cfg, "head")

        # Verify tables created
        engine = create_engine(db_url)
        with engine.connect() as conn:
            from sqlalchemy import inspect
            inspector = inspect(conn)
            tables = inspector.get_table_names()
            assert "enhancement_sessions" in tables
            assert "clarification_rounds" in tables
            assert "analysis_snapshots" in tables
            assert "enhancement_attempts" in tables
            assert "validation_snapshots" in tables
            assert "quality_snapshots" in tables
            assert "feedback" in tables

        # Run downgrade to base
        command.downgrade(alembic_cfg, "base")
    finally:
        if os.path.exists(tmp_db_path):
            try:
                os.remove(tmp_db_path)
            except OSError:
                pass


# ==============================================================================
# TEST GROUP B — FIRST SESSION
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_b_first_session(mock_enhance, client, test_db):
    """Verify first enhance request creates session, returns session_id, and stores original prompt."""
    analysis = make_analysis("build a rest api")
    validation = make_validation(True)
    quality = make_quality(9.0)

    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "Enhanced prompt text",
        validation=validation,
        repair_attempted=False,
        quality=quality,
        initial_enhanced_prompt="Enhanced prompt text",
        initial_validation=validation,
    )

    resp = client.post("/api/enhance", json={"prompt": "build a rest api"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "enhanced"
    assert "session_id" in data
    assert data["history_saved"] is True

    # Check database
    repo = EnhancementSessionRepository(test_db)
    session = repo.get_session(data["session_id"])
    assert session is not None
    assert session.original_prompt == "build a rest api"
    assert session.status == "enhanced"


# ==============================================================================
# TEST GROUP C — CLEAR SUCCESS
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_c_clear_success(mock_enhance, client, test_db):
    """Verify clear successful enhancement persists session, analysis, attempt, validation, quality."""
    analysis = make_analysis("clear coding request", category="CODING")
    validation = make_validation(True)
    quality = make_quality(94)

    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "Enhanced clear prompt",
        validation=validation,
        repair_attempted=False,
        quality=quality,
        initial_enhanced_prompt="Enhanced clear prompt",
        initial_validation=validation,
    )

    resp = client.post("/api/enhance", json={"prompt": "clear coding request"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    repo = EnhancementSessionRepository(test_db)
    session = repo.get_session(session_id)
    assert session is not None
    assert len(session.analysis_snapshots) == 1
    assert session.analysis_snapshots[0].classification == "CODING"
    assert len(session.enhancement_attempts) == 1
    assert session.enhancement_attempts[0].attempt_type == "INITIAL"
    assert session.enhancement_attempts[0].validation is not None
    assert session.enhancement_attempts[0].validation.is_valid is True
    assert session.enhancement_attempts[0].quality is not None
    assert session.enhancement_attempts[0].quality.overall_score == 94.0
    assert session.final_quality_score == 94.0


# ==============================================================================
# TEST GROUP D — CLARIFICATION ROUND
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_d_clarification_round(mock_enhance, client, test_db):
    """Verify multi-round clarification maintains the same persistent session and preserves inputs."""
    # Round 1: Ambiguous prompt returns needs_clarification
    analysis_r1 = make_analysis("make a website", needs_clarification=True)
    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis_r1,
        "",
        validation=None,
        repair_attempted=False,
        quality=None,
    )

    resp1 = client.post("/api/enhance", json={"prompt": "make a website"})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] == "needs_clarification"
    session_id = data1["session_id"]

    # Verify session status and question stored
    repo = EnhancementSessionRepository(test_db)
    session = repo.get_session(session_id)
    assert session.status == "needs_clarification"
    assert len(session.clarification_rounds) == 1
    assert session.clarification_rounds[0].questions_json is not None

    # Round 2: Follow-up answers submitted with same session_id
    analysis_r2 = make_analysis("make a website", needs_clarification=False)
    validation_r2 = make_validation(True)
    quality_r2 = make_quality(91)
    mock_enhance.return_value = EnhanceResult(
        analysis_r2,
        "Enhanced landing page prompt",
        validation=validation_r2,
        repair_attempted=False,
        quality=quality_r2,
        initial_enhanced_prompt="Enhanced landing page prompt",
        initial_validation=validation_r2,
    )

    answers = [
        {"question_id": "q1", "question": "What language?", "answer": "TypeScript with Next.js"}
    ]
    resp2 = client.post(
        "/api/enhance",
        json={
            "session_id": session_id,
            "prompt": "make a website",
            "clarification_answers": answers,
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "enhanced"
    assert data2["session_id"] == session_id

    # Refresh session
    test_db.refresh(session)
    assert session.status == "enhanced"
    # Original prompt unchanged!
    assert session.original_prompt == "make a website"
    # Clarification answers stored in clarification round
    rounds = session.clarification_rounds
    assert len(rounds) >= 2
    # Check that answers_json was recorded
    answer_rounds = [r for r in rounds if r.answers_json is not None]
    assert len(answer_rounds) >= 1
    assert answer_rounds[0].answers_json[0]["answer"] == "TypeScript with Next.js"
    # Multiple analysis snapshots versioned by round
    assert len(session.analysis_snapshots) == 2


# ==============================================================================
# TEST GROUP E — ANALYSIS FAILURE
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_e_analysis_failure(mock_enhance, client, test_db):
    """Verify analysis_failed status persists correctly with no attempts/validation/quality."""
    analysis = make_analysis("gibberish", is_grounded=False)
    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "",
        validation=None,
        repair_attempted=False,
        quality=None,
    )

    resp = client.post("/api/enhance", json={"prompt": "gibberish"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "analysis_failed"
    session_id = data["session_id"]

    repo = EnhancementSessionRepository(test_db)
    session = repo.get_session(session_id)
    assert session.status == "analysis_failed"
    assert len(session.enhancement_attempts) == 0


# ==============================================================================
# TEST GROUP F — VALIDATION REPAIR
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_f_validation_repair(mock_enhance, client, test_db):
    """Verify when validation fails and repair succeeds: both attempts and validations are stored."""
    analysis = make_analysis("repair test prompt")
    val_fail = make_validation(is_valid=False)
    val_pass = make_validation(is_valid=True)
    quality = make_quality(88)

    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "Repaired enhanced prompt text",
        validation=val_pass,
        repair_attempted=True,
        quality=quality,
        initial_enhanced_prompt="Initial bad enhanced prompt text",
        initial_validation=val_fail,
    )

    resp = client.post("/api/enhance", json={"prompt": "repair test prompt"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    repo = EnhancementSessionRepository(test_db)
    session = repo.get_session(session_id)
    assert session.status == "enhanced"
    assert len(session.enhancement_attempts) == 2

    att1 = session.enhancement_attempts[0]
    assert att1.attempt_type == "INITIAL"
    assert att1.validation.is_valid is False
    assert att1.quality is None

    att2 = session.enhancement_attempts[1]
    assert att2.attempt_type == "REPAIR"
    assert att2.validation.is_valid is True
    assert att2.quality is not None
    assert att2.quality.overall_score == 88.0


# ==============================================================================
# TEST GROUP G — VALIDATION FAILURE
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_g_validation_failure(mock_enhance, client, test_db):
    """Verify when repair validation also fails: session status is validation_failed and no quality snapshot."""
    analysis = make_analysis("irreparable prompt")
    val_fail1 = make_validation(is_valid=False)
    val_fail2 = make_validation(is_valid=False)

    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "Failed repaired prompt",
        validation=val_fail2,
        repair_attempted=True,
        quality=None,
        initial_enhanced_prompt="Failed initial prompt",
        initial_validation=val_fail1,
    )

    resp = client.post("/api/enhance", json={"prompt": "irreparable prompt"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "validation_failed"
    session_id = data["session_id"]

    repo = EnhancementSessionRepository(test_db)
    session = repo.get_session(session_id)
    assert session.status == "validation_failed"
    assert len(session.enhancement_attempts) == 2
    assert session.enhancement_attempts[0].quality is None
    assert session.enhancement_attempts[1].quality is None
    assert session.final_quality_score is None


# ==============================================================================
# TEST GROUP H — HISTORY LIST
# ==============================================================================

def test_group_h_history_list(client, test_db):
    """Test history list: pagination, ordering newest first, limit enforcement, summary fields."""
    repo = EnhancementSessionRepository(test_db)

    # Insert 5 test sessions
    for i in range(5):
        s, _ = repo.get_or_create_session(None, f"Prompt number {i}")
        repo.update_session_outcome(
            session=s,
            status="enhanced",
            final_category="CODING",
            final_enhanced_prompt=f"Enhanced {i}",
            final_quality_score=80.0 + (i * 2.0),
        )
    test_db.commit()

    resp = client.get("/api/history?limit=3&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["items"]) == 3
    assert data["limit"] == 3
    assert data["offset"] == 0

    # Test limit clamping
    resp_clamped = client.get("/api/history?limit=200")
    assert resp_clamped.status_code == 200
    assert resp_clamped.json()["limit"] == 100


# ==============================================================================
# TEST GROUP I — HISTORY DETAIL
# ==============================================================================

def test_group_i_history_detail(client, test_db):
    """Test history detail reconstruction without LLM calls."""
    repo = EnhancementSessionRepository(test_db)
    session, _ = repo.get_or_create_session(None, "Detail original prompt")
    analysis = make_analysis("Detail original prompt", category="RESEARCH")
    repo.record_analysis_snapshot(session, analysis)
    val = make_validation(True)
    qual = make_quality(95)
    repo.record_enhancement_attempt(
        session=session,
        attempt_number=1,
        attempt_type="INITIAL",
        enhanced_prompt="Reconstructed enhanced output",
        validation=val,
        quality=qual,
    )
    repo.update_session_outcome(
        session=session,
        status="enhanced",
        final_category="RESEARCH",
        final_enhanced_prompt="Reconstructed enhanced output",
        final_quality_score=95.0,
    )
    test_db.commit()

    resp = client.get(f"/api/history/{session.id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["session_id"] == session.id
    assert detail["original_prompt"] == "Detail original prompt"
    assert detail["status"] == "enhanced"
    assert detail["final_enhanced_prompt"] == "Reconstructed enhanced output"
    assert detail["quality"]["overall_score"] == 95
    assert detail["latest_analysis"]["objective"] == "Objective for Detail original prompt"


# ==============================================================================
# TEST GROUP J — DELETE
# ==============================================================================

def test_group_j_delete_session(client, test_db):
    """Test deleting one session cascades child records and leaves other sessions intact."""
    repo = EnhancementSessionRepository(test_db)
    s1, _ = repo.get_or_create_session(None, "Prompt to delete")
    s2, _ = repo.get_or_create_session(None, "Prompt to keep")
    test_db.commit()

    resp = client.delete(f"/api/history/{s1.id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    assert repo.get_session(s1.id) is None
    assert repo.get_session(s2.id) is not None


# ==============================================================================
# TEST GROUP K — CLEAR HISTORY
# ==============================================================================

def test_group_k_clear_history(client, test_db):
    """Test clearing all history safely removes all sessions."""
    repo = EnhancementSessionRepository(test_db)
    for i in range(3):
        repo.get_or_create_session(None, f"Prompt {i}")
    test_db.commit()

    resp = client.delete("/api/history")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"
    assert resp.json()["deleted_count"] >= 3

    summaries, total = repo.list_history_summaries()
    assert total == 0


# ==============================================================================
# TEST GROUP L & M — FEEDBACK & UPSERT
# ==============================================================================

def test_group_l_and_m_feedback_and_upsert(client, test_db):
    """Test submitting feedback, optional fields, and idempotent upserting."""
    repo = EnhancementSessionRepository(test_db)
    session, _ = repo.get_or_create_session(None, "Feedback target prompt")
    test_db.commit()

    # Initial submission: HELPFUL
    resp1 = client.post(
        "/api/feedback",
        json={
            "session_id": session.id,
            "rating": "HELPFUL",
            "reason": "Great structure",
            "dataset_consent": False,
        },
    )
    assert resp1.status_code == 200
    fb_data = resp1.json()
    assert fb_data["rating"] == "HELPFUL"
    assert fb_data["reason"] == "Great structure"
    assert fb_data["dataset_consent"] is False

    # Upsert / revision: User changes rating to NOT_HELPFUL with edited prompt
    resp2 = client.post(
        "/api/feedback",
        json={
            "session_id": session.id,
            "rating": "NOT_HELPFUL",
            "reason": "Too verbose",
            "edited_prompt": "My custom edited prompt",
            "dataset_consent": True,
        },
    )
    assert resp2.status_code == 200
    fb_data2 = resp2.json()
    assert fb_data2["id"] == fb_data["id"]  # Same record ID!
    assert fb_data2["rating"] == "NOT_HELPFUL"
    assert fb_data2["edited_prompt"] == "My custom edited prompt"
    assert fb_data2["dataset_consent"] is True

    # Check database: exactly one feedback record
    fb_repo = FeedbackRepository(test_db)
    stored_fb = fb_repo.get_feedback(session.id)
    assert stored_fb is not None
    assert stored_fb.rating == "NOT_HELPFUL"
    assert stored_fb.edited_prompt == "My custom edited prompt"


# ==============================================================================
# TEST GROUP N — CONSENT SEPARATION
# ==============================================================================

def test_group_n_consent_separation(client, test_db):
    """Verify local history does NOT infer dataset consent. Default must be False."""
    repo = EnhancementSessionRepository(test_db)
    session, _ = repo.get_or_create_session(None, "Consent prompt")
    test_db.commit()

    # Session created with default False consent
    assert session.dataset_consent is False

    # Feedback submitted without consent field -> still False
    resp = client.post(
        "/api/feedback",
        json={"session_id": session.id, "rating": "HELPFUL"},
    )
    assert resp.status_code == 200
    assert resp.json()["dataset_consent"] is False

    test_db.refresh(session)
    assert session.dataset_consent is False


# ==============================================================================
# TEST GROUP O — HISTORY DISABLED
# ==============================================================================

@patch("app.api.routes.settings.history_enabled", False)
@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_o_history_disabled(mock_enhance, client, test_db):
    """Verify HISTORY_ENABLED=false keeps AI pipeline functional without saving to database."""
    analysis = make_analysis("disabled history prompt")
    val = make_validation(True)
    qual = make_quality(9.0)

    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "Enhanced prompt text",
        validation=val,
        repair_attempted=False,
        quality=qual,
        initial_enhanced_prompt="Enhanced prompt text",
        initial_validation=val,
    )

    resp = client.post("/api/enhance", json={"prompt": "disabled history prompt"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "enhanced"
    assert data["history_saved"] is False

    # Database must have 0 sessions
    repo = EnhancementSessionRepository(test_db)
    summaries, total = repo.list_history_summaries()
    assert total == 0


# ==============================================================================
# TEST GROUP P — PERSISTENCE FAILURE TOLERANCE
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
@patch.object(EnhancementSessionRepository, "update_session_outcome")
def test_group_p_persistence_failure(mock_update, mock_enhance, client):
    """Verify database write failure does NOT corrupt AI output and returns history_saved=False."""
    mock_update.side_effect = RuntimeError("Simulated SQLite write error")

    analysis = make_analysis("failure tolerance prompt")
    val = make_validation(True)
    qual = make_quality(9.0)

    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "Enhanced text despite DB error",
        validation=val,
        repair_attempted=False,
        quality=qual,
        initial_enhanced_prompt="Enhanced text despite DB error",
        initial_validation=val,
    )

    resp = client.post("/api/enhance", json={"prompt": "failure tolerance prompt"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "enhanced"
    assert data["enhanced_prompt"] == "Enhanced text despite DB error"
    assert data["history_saved"] is False


# ==============================================================================
# TEST GROUP Q — ZERO LLM HISTORY CALLS
# ==============================================================================

@patch("app.api.routes.prompt_analyzer.analyze")
@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_q_zero_llm_history(mock_enhance, mock_analyze, client, test_db):
    """Verify history list, detail, delete, and feedback operations make ZERO LLM calls."""
    repo = EnhancementSessionRepository(test_db)
    s, _ = repo.get_or_create_session(None, "Zero LLM test")
    test_db.commit()

    # 1. History list
    client.get("/api/history")
    # 2. History detail
    client.get(f"/api/history/{s.id}")
    # 3. Feedback
    client.post("/api/feedback", json={"session_id": s.id, "rating": "HELPFUL"})
    # 4. History delete
    client.delete(f"/api/history/{s.id}")

    mock_enhance.assert_not_called()
    mock_analyze.assert_not_called()


# ==============================================================================
# TEST GROUP R — AUTHORITATIVE VS DERIVED STORAGE
# ==============================================================================

@patch("app.api.routes.prompt_enhancer.enhance")
def test_group_r_authoritative_vs_derived_storage(mock_enhance, client, test_db):
    """Verify original prompt is never overwritten by enhanced text or clarification answers."""
    analysis = make_analysis("User Authoritative Original Prompt")
    val = make_validation(True)

    from app.core.enhancer import EnhanceResult
    mock_enhance.return_value = EnhanceResult(
        analysis,
        "Completely different derived enhanced prompt text",
        validation=val,
        repair_attempted=False,
        quality=make_quality(9.0),
        initial_enhanced_prompt="Completely different derived enhanced prompt text",
        initial_validation=val,
    )

    resp = client.post("/api/enhance", json={"prompt": "User Authoritative Original Prompt"})
    session_id = resp.json()["session_id"]

    repo = EnhancementSessionRepository(test_db)
    session = repo.get_session(session_id)

    # Authoritative prompt remains exact
    assert session.original_prompt == "User Authoritative Original Prompt"
    # Derived prompt stored in separate column
    assert session.final_enhanced_prompt == "Completely different derived enhanced prompt text"
    assert session.original_prompt != session.final_enhanced_prompt
