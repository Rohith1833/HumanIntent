"""
Production Hardening Regression Test Suite.
Validates the general semantic invariants before Phase 11 training:
1. Vague artifact request: no domain invention, ambiguity high, clarification required, enhancement halted.
2. Vague app request: same general behavior.
3. Sequential-request contamination: zero cross-request domain contamination.
4. Fresh-session isolation: equivalent clean state across sessions.
5. Compound requirement preservation: explicit deliverables, lifecycle deletion constraints, and missing context preserved simultaneously without replacement.
6. Negative constraint preservation.
7. Quantity preservation.
8. Self-correction / supersession preservation.
9. Compound enhancement survives validator when requirements and constraints are preserved.
10. Validator rejects deliberately corrupted enhancement (invented feature, dropped constraint, changed task).
11. Production purity audit: zero diagnostic probe nouns in production code.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.core.analyzer import PromptAnalyzer
from app.core.enhancer import PromptEnhancer
from app.core.validator import PromptValidator
from app.main import app
from app.schemas.analysis import (
    AmbiguityAssessment,
    AmbiguityLevel,
    ClarificationAnswer,
    ClarificationQuestion,
    DeliverableSemantics,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
)
from app.schemas.validation import ValidationIssueType, ValidationSeverity


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# TEST 1 & 2: VAGUE ARTIFACT AND APP REQUESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_vague_system_request_requires_clarification_no_invention():
    """1. Vague artifact request: 'build me a system' must not invent domains and must halt for clarification."""
    mock_client = AsyncMock()
    # LLM attempts to return generic or hallucinated content
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.8},
        "objective": "Build a system",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["application interface type", "data storage method"],
        "ambiguity": {
            "score": 80,
            "level": "HIGH",
            "needs_clarification": True,
            "questions": [],
        },
    })
    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze("build me a system")

    # Invariants
    assert analysis.ambiguity.level != AmbiguityLevel.LOW
    assert analysis.ambiguity.score >= 65
    assert analysis.ambiguity.needs_clarification is True
    assert len(analysis.ambiguity.questions) >= 1
    assert any("purpose" in q.target_dimension.lower() for q in analysis.ambiguity.questions)

    # Must not contain invented domain
    assert not any(d in analysis.objective.lower() for d in ["attendance", "student", "hackathon", "medical", "ecommerce"])
    assert len(analysis.explicit_requirements) == 0


@pytest.mark.asyncio
async def test_vague_app_request_requires_clarification():
    """2. Vague app request: 'make me an app' must halt for clarification."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.8},
        "objective": "Make an application",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["application interface type"],
        "ambiguity": {
            "score": 75,
            "level": "HIGH",
            "needs_clarification": True,
            "questions": [],
        },
    })
    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze("make me an app")

    assert analysis.ambiguity.level == AmbiguityLevel.HIGH
    assert analysis.ambiguity.needs_clarification is True
    assert len(analysis.ambiguity.questions) >= 1
    assert any("purpose" in q.target_dimension.lower() for q in analysis.ambiguity.questions)


@pytest.mark.asyncio
async def test_vague_deliverable_with_audience_only_requires_clarification():
    """Vague deliverable with only recipient/audience ('create something for clients') must halt."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "GENERAL", "confidence": 0.7},
        "objective": "Create something for clients",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["functional purpose"],
        "ambiguity": {
            "score": 85,
            "level": "HIGH",
            "needs_clarification": True,
            "questions": [],
        },
    })
    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze("create something for clients")

    assert analysis.ambiguity.level == AmbiguityLevel.HIGH
    assert analysis.ambiguity.needs_clarification is True
    assert len(analysis.ambiguity.questions) >= 1


# ==============================================================================
# TEST 3 & 4: SEQUENTIAL CONTAMINATION AND FRESH-SESSION ISOLATION
# ==============================================================================

def test_sequential_requests_zero_cross_contamination(client):
    """3. Sequential requests: Request B must contain zero domain details from Request A."""
    analysis_a_json = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.9},
        "objective": "Build an employee attendance tracking portal",
        "explicit_requirements": ["employee attendance", "portal"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["data storage method"],
        "ambiguity": {"score": 20, "level": "LOW", "needs_clarification": False, "questions": []},
    })
    analysis_b_json = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.8},
        "objective": "Build a system",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["application interface type"],
        "ambiguity": {"score": 80, "level": "HIGH", "needs_clarification": True, "questions": []},
    })
    validator_json = json.dumps({"is_valid": True, "issues": []})
    quality_json = json.dumps({
        "clarity": {"score": 90, "reason": "Clear"},
        "specificity": {"score": 90, "reason": "Specific"},
        "output_definition": {"score": 90, "reason": "Defined"},
        "structure": {"score": 90, "reason": "Structured"},
        "concision": {"score": 90, "reason": "Concise"},
    })

    with patch("app.llm.ollama_client.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = [
            analysis_a_json,
            "# Enhanced Prompt\nBuild an employee attendance tracking portal with clean design.",
            validator_json,
            quality_json,
            analysis_b_json,
        ]

        res_a = client.post("/api/enhance", json={"prompt": "build an employee attendance tracking portal"})
        assert res_a.status_code == 200
        data_a = res_a.json()
        assert data_a["status"] == "enhanced"
        session_a_id = data_a["session_id"]

        # Second: Generic request passed sequentially with session_id from Request A
        res_b = client.post("/api/enhance", json={
            "prompt": "build me a system",
            "session_id": session_a_id,  # Attempt cross-session reuse
        })
        assert res_b.status_code == 200
        data_b = res_b.json()
        # Session prompt mismatch forces new isolated session and triggers clarification
        assert data_b["status"] == "needs_clarification"
        assert data_b["session_id"] != session_a_id
        # Zero semantic contamination
        analysis_b = data_b["analysis"]
        assert "attendance" not in analysis_b["objective"].lower()
        assert not any("attendance" in req.lower() for req in analysis_b["explicit_requirements"])
        assert not any("attendance" in q["question"].lower() for q in data_b["questions"])


def test_fresh_session_isolation_consistency(client):
    """4. Fresh-session isolation: A request in a fresh session yields the same clean result."""
    analysis_json = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.8},
        "objective": "Build a system",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["application interface type"],
        "ambiguity": {"score": 80, "level": "HIGH", "needs_clarification": True, "questions": []},
    })
    with patch("app.llm.ollama_client.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = analysis_json

        res = client.post("/api/enhance", json={"prompt": "build me a system"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "needs_clarification"
        assert "CRITICAL_AMBIGUITY" in data["reason_codes"]


# ==============================================================================
# TEST 5: COMPOUND REQUIREMENT & LIFECYCLE CONSTRAINT PRESERVATION
# ==============================================================================

@pytest.mark.asyncio
async def test_compound_preservation_lifecycle_and_missing_context():
    """5. Compound prompt: deliverables, lifecycle deletion constraint, and unspecified storage coexist."""
    compound_prompt = (
        "Build a multi-user conference platform with user registration, session scheduling, "
        "and presentation upload. Deciding where data should be stored is up to you. "
        "All participant data must be deleted after the conference finishes."
    )
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a multi-user conference platform",
        "explicit_requirements": ["user registration", "session scheduling", "presentation upload"],
        "constraints": ["delete participant data after the conference finishes"],
        "preferences": [],
        "missing_context": ["data storage method"],
        "ambiguity": {"score": 25, "level": "LOW", "needs_clarification": False, "questions": []},
    })
    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(compound_prompt)

    # Deliverables preserved
    reqs_text = " ".join(analysis.explicit_requirements).lower()
    assert "registration" in reqs_text
    assert "scheduling" in reqs_text
    assert "upload" in reqs_text

    # Lifecycle constraint preserved under CONSTRAINTS
    constraints_text = " ".join(analysis.constraints).lower()
    assert any(term in constraints_text for term in ["delete", "deleted", "after the conference"])

    # Unspecified dimension preserved under MISSING_CONTEXT
    missing_text = " ".join(analysis.missing_context).lower()
    assert "storage" in missing_text or "data storage" in missing_text

    # Neither replaces the other
    assert len(analysis.constraints) >= 1
    assert len(analysis.missing_context) >= 1


# ==============================================================================
# TEST 6: NEGATIVE CONSTRAINT PRESERVATION
# ==============================================================================

@pytest.mark.asyncio
async def test_negative_constraint_preservation():
    """6. Negative constraints (e.g. 'do not use external libraries') must be preserved in constraints."""
    prompt = "Write a Python script to parse CSV files without external libraries"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Parse CSV files in Python",
        "explicit_requirements": ["Python", "parse CSV files"],
        "constraints": ["do not use external libraries"],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {"score": 15, "level": "LOW", "needs_clarification": False, "questions": []},
    })
    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(prompt)

    constraints_text = " ".join(analysis.constraints).lower()
    assert "without" in constraints_text or "do not use" in constraints_text or "external libraries" in constraints_text


# ==============================================================================
# TEST 7: QUANTITY PRESERVATION
# ==============================================================================

@pytest.mark.asyncio
async def test_quantity_preservation():
    """7. Quantities in prompts must be preserved in explicit requirements."""
    prompt = "Give me 4 project ideas and 3 interview questions for data engineering"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "GENERAL", "confidence": 0.9},
        "objective": "Provide project ideas and interview questions for data engineering",
        "explicit_requirements": ["4 project ideas", "3 interview questions"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {"score": 10, "level": "LOW", "needs_clarification": False, "questions": []},
    })
    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(prompt)

    reqs_text = " ".join(analysis.explicit_requirements).lower()
    assert "4" in reqs_text
    assert "3" in reqs_text


# ==============================================================================
# TEST 8: SELF-CORRECTION / SUPERSESSION PRESERVATION
# ==============================================================================

@pytest.mark.asyncio
async def test_supersession_rejection():
    """8. Superseded requirements ('actually scratch that, use Postgres') must discard rejected option."""
    prompt = "Create a database schema using MySQL, actually scratch that, use PostgreSQL instead"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Create database schema with PostgreSQL",
        "explicit_requirements": ["PostgreSQL"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {"score": 10, "level": "LOW", "needs_clarification": False, "questions": []},
    })
    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(prompt)

    reqs_text = " ".join(analysis.explicit_requirements).lower()
    assert "postgresql" in reqs_text or "postgres" in reqs_text
    assert "mysql" not in reqs_text


# ==============================================================================
# TEST 9: COMPOUND ENHANCEMENT SURVIVES VALIDATOR
# ==============================================================================

@pytest.mark.asyncio
async def test_compound_enhancement_survives_validator():
    """9. Compound enhancement survives validator when all requirements and constraints are preserved."""
    original_prompt = (
        "Build a notification service that sends email alerts, logs all events, "
        "and purges logs after 30 days. Implementation language is your choice."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build a notification service with alerts, logging, and automatic purging",
        explicit_requirements=["email alerts", "event logging"],
        constraints=["purge logs after 30 days"],
        preferences=[],
        missing_context=["implementation language"],
        ambiguity=AmbiguityAssessment(score=20, level=AmbiguityLevel.LOW, needs_clarification=False),
    )
    enhanced_prompt = (
        "# NOTIFICATION SERVICE SPECIFICATION\n\n"
        "## OBJECTIVE\n"
        "Build a robust notification service that manages automated alerts and event auditing.\n\n"
        "## EXPLICIT REQUIREMENTS\n"
        "- Email Alerts: Send timely email notifications to configured recipients.\n"
        "- Event Logging: Record detailed event logs for each notification attempt.\n\n"
        "## CONSTRAINTS\n"
        "- Retention Policy: Automatically purge logs after 30 days.\n\n"
        "## UNSPECIFIED DETAILS\n"
        "- Implementation Language: Select an appropriate, high-performance programming language."
    )
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_client)
    report = await validator.validate(
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        analysis=analysis,
    )
    assert report.is_valid is True
    assert len(report.issues) == 0


# ==============================================================================
# TEST 10: VALIDATOR REJECTS DELIBERATELY CORRUPTED ENHANCEMENT
# ==============================================================================

@pytest.mark.asyncio
async def test_validator_rejects_corrupted_enhancement():
    """10. Validator rejects enhancements that invent unrequested features or drop constraints."""
    original_prompt = "Write a python script to resize images. Keep files under 2MB."
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Resize images in Python under 2MB",
        explicit_requirements=["Python", "resize images"],
        constraints=["under 2MB"],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(score=10, level=AmbiguityLevel.LOW, needs_clarification=False),
    )
    # Corrupted: dropped constraint under 2MB, invented AWS S3 and Stripe payments
    corrupted_enhanced = (
        "# Enhanced Prompt\n"
        "Build an image resizing tool connected to AWS S3 and integrated with Stripe subscriptions."
    )
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": False,
        "issues": [
            {
                "type": "MISSING_CONSTRAINT",
                "severity": "ERROR",
                "message": "Dropped mandatory constraint 'under 2MB'.",
                "evidence": "2MB",
            },
            {
                "type": "INVENTED_FEATURE",
                "severity": "ERROR",
                "message": "Invented unrequested payment processing with Stripe.",
                "evidence": "Stripe",
            },
        ],
    })
    validator = PromptValidator(client=mock_client)
    report = await validator.validate(
        original_prompt=original_prompt,
        enhanced_prompt=corrupted_enhanced,
        analysis=analysis,
    )
    assert report.is_valid is False
    assert any(i.severity == ValidationSeverity.ERROR for i in report.issues)


# ==============================================================================
# TEST 11: PRODUCTION PURITY AUDIT
# ==============================================================================

def test_production_purity_audit():
    """
    11. Production purity audit:
    Assert zero occurrences of diagnostic probe nouns (attendance, hackathon, students)
    in production code (backend/app/**/*.py).
    """
    app_dir = Path(__file__).resolve().parent.parent / "app"
    forbidden_nouns = ["attendance", "hackathon", "student", "students"]

    violations = []
    for py_file in app_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8").lower()
        for noun in forbidden_nouns:
            if noun in content:
                violations.append(f"{py_file.name}: contains forbidden probe noun '{noun}'")

    assert len(violations) == 0, f"Production purity violations found: {violations}"
