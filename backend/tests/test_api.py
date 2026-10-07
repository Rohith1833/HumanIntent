import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest

from app.core.analyzer import PromptAnalyzer
from app.core.enhancer import PromptEnhancer
from app.llm.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.main import app
from app.schemas.analysis import (
    AmbiguityAssessment,
    AmbiguityLevel,
    AnalysisGrounding,
    ClarificationAnswer,
    ClarificationQuestion,
    DeliverableSemantics,
    GroundingEvidence,
    GroundingRelation,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
    SourceType,
)
from app.templates.base import BaseTemplate
from app.templates.selector import get_template
from app.templates import (
    CodingTemplate,
    DebuggingTemplate,
    ResearchTemplate,
    EducationTemplate,
    WritingTemplate,
    DataAnalysisTemplate,
    PlanningTemplate,
    CreativeTemplate,
    ImageGenerationTemplate,
    SummarizationTemplate,
    GeneralTemplate,
)
from app.core.validator import PromptValidator
from app.schemas.validation import (
    PromptValidation,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
)
from app.core.scorer import PromptScorer
from app.schemas.quality import (
    InflationRisk,
    PromptQualityReport,
    QualityDimension,
    QUALITY_WEIGHTS,
    get_score_label,
)

client = TestClient(app)


# --- Root & Health Tests ---

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "HumanIntent API"


def test_health_endpoint_ollama_reachable():
    with patch("app.api.routes.ollama_client.check_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = True
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ollama"]["reachable"] is True
        assert "model" in data["ollama"]


def test_health_endpoint_ollama_unreachable():
    with patch("app.api.routes.ollama_client.check_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = False
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ollama"]["reachable"] is False


# --- Model Listing Tests ---

def test_list_models_success():
    with patch("app.api.routes.ollama_client.list_models", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = ["llama3:latest", "llama3.2:latest"]
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "llama3:latest" in data["models"]
        assert data["current_model"] is not None


def test_list_models_ollama_offline():
    with patch("app.api.routes.ollama_client.list_models", new_callable=AsyncMock) as mock_list:
        mock_list.side_effect = OllamaConnectionError("Could not connect to Ollama.")
        response = client.get("/api/models")
        assert response.status_code == 503


# --- Standalone /api/analyze Tests ---

def test_analyze_rejects_empty_prompt():
    response = client.post("/api/analyze", json={"prompt": ""})
    assert response.status_code == 422


def test_analyze_rejects_whitespace_prompt():
    response = client.post("/api/analyze", json={"prompt": "   \n\t  "})
    assert response.status_code == 422


def test_analyze_success():
    mock_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.96),
        objective="Build a college attendance system",
        explicit_requirements=["Python", "college attendance"],
        constraints=[],
        preferences=[],
        missing_context=["application interface type", "data storage method"],
    )
    with patch("app.api.routes.prompt_analyzer.analyze", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = mock_analysis
        response = client.post(
            "/api/analyze",
            json={"prompt": "make python attendance app for college"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["classification"]["category"] == "CODING"
        assert data["classification"]["confidence"] == 0.96
        assert data["objective"] == "Build a college attendance system"
        assert "Python" in data["explicit_requirements"]
        assert "data storage method" in data["missing_context"]


# --- Analyzer Service Logic & Structured Output Parsing Tests ---

@pytest.mark.asyncio
async def test_analyzer_parsing_success():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build attendance app",
        "explicit_requirements": ["Python"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["data storage method", "authentication requirements"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make python attendance app")

    assert isinstance(result, PromptAnalysis)
    assert result.classification.category == PromptCategory.CODING
    assert result.classification.confidence == 0.95
    assert result.objective == "Build attendance app"
    assert result.explicit_requirements == ["Python"]
    assert "data storage method" in result.missing_context


@pytest.mark.asyncio
async def test_analyzer_strips_markdown_fences():
    mock_client = AsyncMock()
    mock_client.chat.return_value = "```json\n" + json.dumps({
        "classification": {"category": "CODING", "confidence": 0.92},
        "objective": "Build attendance app",
        "explicit_requirements": ["Python"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    }) + "\n```"

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make python attendance app")
    assert result.classification.category == PromptCategory.CODING
    assert result.objective == "Build attendance app"


@pytest.mark.asyncio
async def test_analyzer_malformed_json_raises():
    mock_client = AsyncMock()
    mock_client.chat.return_value = "This is not valid json at all"

    analyzer = PromptAnalyzer(client=mock_client)
    with pytest.raises(OllamaResponseError, match="invalid JSON"):
        await analyzer.analyze("make python attendance app")


@pytest.mark.asyncio
async def test_analyzer_missing_required_field_raises():
    mock_client = AsyncMock()
    # Missing required 'objective' and 'classification' fields
    mock_client.chat.return_value = json.dumps({
        "explicit_requirements": ["Python"],
        "constraints": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    with pytest.raises(OllamaResponseError, match="does not match the required schema"):
        await analyzer.analyze("make python attendance app")


# --- Phase 3: Classification Validation & Category Tests ---

@pytest.mark.asyncio
async def test_classification_unknown_category_rejected():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "FINANCE_MAGIC", "confidence": 0.9},
        "objective": "Do finance magic",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    with pytest.raises(OllamaResponseError, match="does not match the required schema"):
        await analyzer.analyze("do finance magic")


@pytest.mark.asyncio
async def test_classification_confidence_out_of_bounds_rejected():
    mock_client = AsyncMock()
    # Test confidence > 1.0
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 1.5},
        "objective": "Build something",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    with pytest.raises(OllamaResponseError, match="does not match the required schema"):
        await analyzer.analyze("build something")

    # Test confidence < 0.0
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": -0.2},
        "objective": "Build something",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    with pytest.raises(OllamaResponseError, match="does not match the required schema"):
        await analyzer.analyze("build something")


@pytest.mark.asyncio
async def test_classification_case_normalization():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "coding", "confidence": 0.95},
        "objective": "Build an attendance app",
        "explicit_requirements": ["Python"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make python attendance app")
    assert result.classification.category == PromptCategory.CODING


@pytest.mark.parametrize("category, prompt_example", [
    (PromptCategory.CODING, "build a Python attendance app"),
    (PromptCategory.DEBUGGING, "my Python API returns a 500 error, fix it"),
    (PromptCategory.EDUCATION, "explain binary search simply"),
    (PromptCategory.RESEARCH, "research blockchain applications in sustainable finance"),
    (PromptCategory.WRITING, "write a professional email to my professor"),
    (PromptCategory.DATA_ANALYSIS, "analyze this CSV and identify sales trends"),
    (PromptCategory.PLANNING, "create a roadmap to learn machine learning"),
    (PromptCategory.CREATIVE, "write a short science fiction story"),
    (PromptCategory.IMAGE_GENERATION, "create an image prompt for a futuristic Indian city"),
    (PromptCategory.SUMMARIZATION, "summarize this article into five points"),
])
@pytest.mark.asyncio
async def test_ten_categories_classification(category: PromptCategory, prompt_example: str):
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": category.value, "confidence": 0.92},
        "objective": f"Handle {prompt_example}",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt_example)
    assert result.classification.category == category
    assert result.classification.confidence == 0.92


# --- Upgraded /api/enhance Tests ---

def test_enhance_rejects_empty_prompt():
    response = client.post("/api/enhance", json={"prompt": ""})
    assert response.status_code == 422


def test_enhance_rejects_whitespace_prompt():
    response = client.post("/api/enhance", json={"prompt": "   \n\t  "})
    assert response.status_code == 422


def test_enhance_returns_both_analysis_and_enhanced_prompt():
    mock_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a college attendance tracker",
        explicit_requirements=["Python", "college attendance"],
        constraints=[],
        preferences=[],
        missing_context=["data storage method", "application interface type"],
    )
    with patch("app.api.routes.prompt_enhancer.enhance", new_callable=AsyncMock) as mock_enhance:
        mock_enhance.return_value = (
            mock_analysis,
            "Create a Python application to track college student attendance.",
        )
        response = client.post(
            "/api/enhance",
            json={"prompt": "make python attendance app for college"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["original_prompt"] == "make python attendance app for college"
        assert "analysis" in data
        assert data["analysis"]["classification"]["category"] == "CODING"
        assert data["analysis"]["classification"]["confidence"] == 0.95
        assert data["analysis"]["objective"] == "Build a college attendance tracker"
        assert "Python" in data["analysis"]["explicit_requirements"]
        assert "data storage method" in data["analysis"]["missing_context"]
        assert "Create a Python application" in data["enhanced_prompt"]
        assert "model" in data


@pytest.mark.asyncio
async def test_enhancer_reuses_analysis_once():
    mock_client = AsyncMock()
    mock_client.chat.return_value = "Enhanced prompt result"

    mock_analyzer = AsyncMock()
    analysis_obj = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.94),
        objective="Build attendance app",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=["data storage method"],
    )
    mock_analyzer.analyze.return_value = analysis_obj

    enhancer = PromptEnhancer(client=mock_client, analyzer=mock_analyzer)
    returned_analysis, enhanced_text = await enhancer.enhance("make python attendance app")

    # Verify analyzer was invoked exactly once (1 analysis LLM call)
    mock_analyzer.analyze.assert_called_once_with("make python attendance app")
    # Verify the message passed to chat contains both the original prompt and analysis
    mock_client.chat.assert_called_once()
    call_args = mock_client.chat.call_args[1]
    messages = call_args["messages"]
    user_msg = messages[1]["content"]
    assert "make python attendance app" in user_msg
    assert "Build attendance app" in user_msg
    assert "explicit_requirements" in user_msg
    assert "CODING" in user_msg
    assert returned_analysis == analysis_obj
    assert enhanced_text == "Enhanced prompt result"


def test_explicit_vs_missing_distinction_preserved():
    mock_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.90),
        objective="Build python attendance app",
        explicit_requirements=["Python"],
        constraints=["offline only"],
        preferences=["simple"],
        missing_context=["data storage method", "application interface type"],
    )
    with patch("app.api.routes.prompt_enhancer.enhance", new_callable=AsyncMock) as mock_enhance:
        mock_enhance.return_value = (mock_analysis, "Prompt here")
        response = client.post("/api/enhance", json={"prompt": "make python attendance app"})
        data = response.json()

        # Explicit requirements must contain what user stated
        assert "Python" in data["analysis"]["explicit_requirements"]
        assert "offline only" in data["analysis"]["constraints"]

        # Missing context must NOT be placed into explicit requirements
        assert "data storage method" not in data["analysis"]["explicit_requirements"]
        assert "application interface type" not in data["analysis"]["explicit_requirements"]
        assert "data storage method" in data["analysis"]["missing_context"]
        assert "application interface type" in data["analysis"]["missing_context"]


# --- Error Propagation Tests ---

def test_enhance_ollama_connection_error():
    with patch("app.api.routes.prompt_enhancer.enhance", new_callable=AsyncMock) as mock_enhance:
        mock_enhance.side_effect = OllamaConnectionError(
            "Could not connect to Ollama at http://localhost:11434. Make sure Ollama is running locally."
        )
        response = client.post("/api/enhance", json={"prompt": "test prompt"})
        assert response.status_code == 503
        assert "Make sure Ollama is running locally" in response.json()["detail"]


def test_enhance_ollama_model_not_found():
    with patch("app.api.routes.prompt_enhancer.enhance", new_callable=AsyncMock) as mock_enhance:
        mock_enhance.side_effect = OllamaModelNotFoundError(
            "Configured model 'llama3.2' is not installed in Ollama."
        )
        response = client.post("/api/enhance", json={"prompt": "test prompt"})
        assert response.status_code == 404
        assert "not installed in Ollama" in response.json()["detail"]


def test_enhance_ollama_timeout():
    with patch("app.api.routes.prompt_enhancer.enhance", new_callable=AsyncMock) as mock_enhance:
        mock_enhance.side_effect = OllamaTimeoutError("Ollama took longer than 120s to respond.")
        response = client.post("/api/enhance", json={"prompt": "test prompt"})
        assert response.status_code == 504
        assert "longer than" in response.json()["detail"] or "timed out" in response.json()["detail"]


# --- Core Intent-Preservation Regression Tests (Phase 2 Repair) ---

@pytest.mark.asyncio
async def test_regression_a_attendance_app_no_invented_technologies():
    """TEST A: Input 'make python attendance app for college' must NOT place invented technologies in explicit requirements."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build an attendance management application for a college",
        "explicit_requirements": ["Python", "college attendance management"],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "application interface type",
            "data storage method",
            "authentication and authorization requirements",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make python attendance app for college")

    assert result.classification.category == PromptCategory.CODING
    assert "Python" in result.explicit_requirements
    forbidden_inventions = [
        "SQLite", "PostgreSQL", "CSV", "React", "FastAPI",
        "Django", "Firebase", "AWS", "CLI", "GUI",
    ]
    for tech in forbidden_inventions:
        assert tech not in result.explicit_requirements

    # Missing context must be neutral dimensions, not candidate suggestions
    for item in result.missing_context:
        assert "SQLite or CSV" not in item
        assert "CLI, desktop GUI, or web" not in item


@pytest.mark.asyncio
async def test_regression_b_missing_context_not_forced_into_enhancement():
    """TEST B: Missing data storage method must not be converted into SQLite, PostgreSQL, MongoDB, or CSV."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "OBJECTIVE:\nCreate an attendance application.\n\n"
        "EXPLICIT REQUIREMENTS:\n- Python\n\n"
        "UNSPECIFIED DETAILS:\n- The user has not specified the data storage method."
    )

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.92),
        objective="Build an attendance application",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=["data storage method"],
    )

    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance("make python attendance app", existing_analysis=analysis)

    call_args = mock_client.chat.call_args[1]
    user_prompt_sent = call_args["messages"][1]["content"]
    assert "Do NOT invent technologies or specifications for any missing context" in user_prompt_sent
    assert "data storage method" in user_prompt_sent

    forbidden_mandates = ["Use SQLite", "Use PostgreSQL", "Use MongoDB", "Use CSV"]
    for mandate in forbidden_mandates:
        assert mandate not in enhanced


@pytest.mark.asyncio
async def test_regression_c_portfolio_no_invented_frameworks():
    """TEST C: Input 'build portfolio website' must NOT invent React, Next.js, Tailwind, Firebase, AWS."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.88},
        "objective": "Build a personal portfolio website",
        "explicit_requirements": ["portfolio website"],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "frontend technology stack",
            "styling and design approach",
            "hosting and deployment platform",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build portfolio website")

    assert result.classification.category == PromptCategory.CODING
    assert "portfolio website" in result.explicit_requirements
    forbidden = ["React", "Next.js", "Tailwind", "Firebase", "AWS"]
    for tech in forbidden:
        assert tech not in result.explicit_requirements


@pytest.mark.asyncio
async def test_regression_d_react_portfolio_preserves_explicit_requirement():
    """TEST D: Input 'build a React portfolio website' must KEEP React as explicit and not remove it."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.94},
        "objective": "Build a portfolio website using React",
        "explicit_requirements": ["React", "portfolio website"],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "styling solution",
            "hosting platform",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build a React portfolio website")

    assert "React" in result.explicit_requirements
    assert "portfolio website" in result.explicit_requirements


@pytest.mark.asyncio
async def test_regression_e_fastapi_and_postgresql_remain_explicit():
    """TEST E: Input 'create a REST API using FastAPI and PostgreSQL' must KEEP FastAPI and PostgreSQL as explicit."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.97},
        "objective": "Create a REST API",
        "explicit_requirements": ["FastAPI", "PostgreSQL", "REST API"],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "authentication method",
            "data domain and schema requirements",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("create a REST API using FastAPI and PostgreSQL")

    assert "FastAPI" in result.explicit_requirements
    assert "PostgreSQL" in result.explicit_requirements
    assert "REST API" in result.explicit_requirements


# --- Semantic Quality Regression Tests (Phase 3 Repair) ---

@pytest.mark.asyncio
async def test_semantic_repair_a_missing_context_does_not_repeat_objective():
    """TEST A: Missing context must NOT repeat the objective, domain, or request words."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a Python attendance management application for a college",
        "explicit_requirements": ["Python"],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "application interface type",
            "data storage method",
            "authentication requirements",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make python attendance app for college")

    forbidden_as_missing = [
        "college attendance system",
        "attendance tracking",
        "python application",
        "college environment",
        "attendance app",
    ]
    for item in result.missing_context:
        lower_item = item.lower()
        for forbidden in forbidden_as_missing:
            assert forbidden != lower_item


def test_semantic_repair_b_concept_not_in_both_objective_and_missing_context():
    """TEST B: Ensure the same semantic concept does not appear in both objective and missing_context."""
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a college attendance management application",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=[
            "application interface type",
            "data storage method",
        ],
    )

    # Missing context must not contain the objective or its primary keywords
    obj_words = set(analysis.objective.lower().split())
    for item in analysis.missing_context:
        assert item.lower() not in analysis.objective.lower()
        # Not a complete repetition of the core subject
        assert "attendance management" not in item.lower()


def test_semantic_repair_c_enhancement_output_never_starts_with_preamble():
    """TEST C: Output cleanup must strip conversational preambles (Here is the enhanced prompt, Sure, Certainly, etc.)."""
    enhancer = PromptEnhancer()

    preamble_cases = [
        ("Here is the enhanced prompt:\n\nAct as a software engineer.", "Act as a software engineer."),
        ("Here's the enhanced prompt:\nAct as a developer.", "Act as a developer."),
        ("Enhanced version:\n\nROLE\nCreate an app.", "ROLE\nCreate an app."),
        ("Sure, here is the enhanced prompt:\n\nAct as an engineer.", "Act as an engineer."),
        ("Certainly! Here is an enhanced version:\n\nAct as an engineer.", "Act as an engineer."),
        ("Act as an experienced Python software developer.", "Act as an experienced Python software developer."),
    ]

    for raw, expected in preamble_cases:
        cleaned = enhancer._clean_enhanced_output(raw)
        assert cleaned == expected
        assert not cleaned.lower().startswith("here is the enhanced prompt")
        assert not cleaned.lower().startswith("sure")
        assert not cleaned.lower().startswith("certainly")


@pytest.mark.asyncio
async def test_semantic_repair_d_educational_prompt_not_exhaustive_checklist():
    """TEST D: For 'explain binary search simply', missing_context should not become an exhaustive checklist."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "EDUCATION", "confidence": 0.99},
        "objective": "Explain the binary search algorithm in simple, beginner-friendly terms",
        "explicit_requirements": ["binary search"],
        "constraints": [],
        "preferences": ["simple explanation"],
        "missing_context": [
            "target audience knowledge level",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("explain binary search simply")

    assert result.classification.category == PromptCategory.EDUCATION
    assert "simple explanation" in result.preferences or "simple" in result.objective.lower()
    # High-impact only: should not be an exhaustive low-level implementation checklist
    assert len(result.missing_context) <= 2
    assert "input data structure" not in result.missing_context


@pytest.mark.asyncio
async def test_semantic_repair_e_professional_email_preferences_and_missing_context():
    """TEST E: For 'write a professional email', recipient and purpose are missing context, professional tone is preference."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "WRITING", "confidence": 0.95},
        "objective": "Draft an email for professional communication",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["professional tone"],
        "missing_context": [
            "recipient",
            "email purpose",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("write a professional email")

    assert result.classification.category == PromptCategory.WRITING
    assert "professional tone" in result.preferences
    assert "recipient" in result.missing_context
    assert "email purpose" in result.missing_context
    # Tone must NOT be considered missing context
    for item in result.missing_context:
        assert "professional" not in item.lower()


@pytest.mark.asyncio
async def test_final_repair_a_empty_list_sentinel_cleanup():
    """TEST A: Empty list sentinels like 'None specified', 'N/A' must normalize to empty lists []."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a portfolio website",
        "explicit_requirements": ["None specified"],
        "constraints": ["N/A"],
        "preferences": ["None"],
        "missing_context": ["No missing context"],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build portfolio website")
    assert result.explicit_requirements == []
    assert result.constraints == []
    assert result.preferences == []
    assert result.missing_context == []


@pytest.mark.asyncio
async def test_final_repair_b_educational_preference_not_technical_requirement():
    """TEST B: 'simply' in 'explain binary search simply' must be a preference, not an explicit requirement."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "EDUCATION", "confidence": 0.98},
        "objective": "Explain the concept of binary search",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["simple explanation"],
        "missing_context": ["target audience knowledge level"],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("explain binary search simply")
    assert result.explicit_requirements == []
    assert any("simple" in p.lower() for p in result.preferences)


@pytest.mark.asyncio
async def test_final_repair_c_professional_email_preferences_and_missing_context():
    """TEST C: 'professional email' -> tone is preference, recipient/purpose is missing context."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "WRITING", "confidence": 0.95},
        "objective": "Draft a professional email",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["professional tone"],
        "missing_context": ["recipient", "email purpose"],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("write a professional email")
    assert any("professional" in p.lower() for p in result.preferences)
    assert "recipient" in result.missing_context
    assert "email purpose" in result.missing_context
    assert result.explicit_requirements == []


@pytest.mark.asyncio
async def test_final_repair_d_portfolio_empty_arrays_never_sentinels():
    """TEST D: 'build portfolio website' has no explicit tech; must normalize to [] not ['None specified']."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a portfolio website",
        "explicit_requirements": ["None specified"],
        "constraints": ["None"],
        "preferences": ["None specified"],
        "missing_context": ["target audience", "portfolio contents"],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build portfolio website")
    assert result.explicit_requirements == []
    assert result.preferences == []
    assert "None specified" not in result.explicit_requirements


@pytest.mark.asyncio
async def test_final_repair_e_explicit_framework_preservation():
    """TEST E: 'build portfolio website using React' -> React must remain an explicit requirement."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a portfolio website",
        "explicit_requirements": ["React"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["target audience"],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build portfolio website using React")
    assert "React" in result.explicit_requirements


@pytest.mark.asyncio
async def test_final_repair_f_direct_downstream_language_no_meta_phrasing():
    """TEST F: Enhanced prompt must not contain meta phrasing like 'Instruct the implementing AI'."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "Instruct the implementing AI:\n"
        "Create a Python attendance app for college.\n\n"
        "OBJECTIVE\nCreate an attendance app.\n\n"
        "EXPLICIT REQUIREMENTS\n- Python\n\n"
        "UNSPECIFIED DETAILS\n"
        "The user has not specified:\n- application interface type\n\n"
        "Do not assume technologies that were not provided."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a Python attendance app for college",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=["application interface type"],
    )
    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance("make python attendance app for college", existing_analysis=analysis)

    forbidden_meta = [
        "Instruct the implementing AI",
        "Tell the AI to",
        "Ask the model to",
    ]
    for meta in forbidden_meta:
        assert meta not in enhanced


@pytest.mark.asyncio
async def test_consistency_a_professional_email_preferences_recovery():
    """TEST A: 'write a professional email' must recover 'professional tone' under preferences."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "WRITING", "confidence": 0.95},
        "objective": "Write a professional email",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],  # Model omitted preference
        "missing_context": ["recipient", "email purpose"],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("write a professional email")
    assert any("professional" in p.lower() for p in result.preferences)


@pytest.mark.asyncio
async def test_consistency_b_professional_email_no_tone_in_missing_context():
    """TEST B: 'write a professional email' must NOT have tone in missing_context."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "WRITING", "confidence": 0.95},
        "objective": "Write a professional email",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["professional tone"],
        "missing_context": ["recipient", "email purpose", "email tone", "professional tone", "tone"],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("write a professional email")
    assert "recipient" in result.missing_context
    assert "email purpose" in result.missing_context
    for item in result.missing_context:
        assert "tone" not in item.lower()


@pytest.mark.asyncio
async def test_consistency_c_attendance_app_no_objective_restatements_in_missing_context():
    """TEST C: 'make python attendance app for college' must NOT restate college attendance as missing context."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Make a Python attendance app for college",
        "explicit_requirements": ["Python"],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "college attendance system",
            "college attendance requirements",
            "attendance application",
            "college attendance system requirements",
            "data storage method",
        ],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make python attendance app for college")
    assert "data storage method" in result.missing_context
    forbidden = [
        "college attendance system",
        "college attendance requirements",
        "attendance application",
        "college attendance system requirements",
    ]
    for bad in forbidden:
        assert bad not in result.missing_context


@pytest.mark.asyncio
async def test_consistency_d_empty_explicit_requirements_no_none_specified():
    """TEST D: When explicit_requirements is empty, enhanced prompt must omit section and not output 'None specified'."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "Write a professional email.\n\n"
        "ROLE & OBJECTIVE\nAct as a professional email writer.\n\n"
        "EXPLICIT REQUIREMENTS\nNone specified.\n\n"
        "UNSPECIFIED DETAILS\n- recipient\n- email purpose\n\n"
        "OUTPUT EXPECTATIONS\nProvide the written email."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.95),
        objective="Write a professional email",
        explicit_requirements=[],
        constraints=[],
        preferences=["professional tone"],
        missing_context=["recipient", "email purpose"],
    )
    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance("write a professional email", existing_analysis=analysis)
    assert "None specified" not in enhanced
    assert "EXPLICIT REQUIREMENTS" not in enhanced


@pytest.mark.asyncio
async def test_consistency_e_preferences_tone_no_tone_in_unspecified_details():
    """TEST E: When preferences contains professional tone, enhanced output must not list tone under unspecified details."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "Write a professional email.\n\n"
        "ROLE & OBJECTIVE\nAct as a professional email writer.\n\n"
        "PREFERENCES\n- Professional tone.\n\n"
        "UNSPECIFIED DETAILS\n"
        "The user has not specified:\n"
        "- Email recipient\n"
        "- Email purpose\n"
        "- Email tone\n\n"
        "OUTPUT EXPECTATIONS\nProvide the written email."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.95),
        objective="Write a professional email",
        explicit_requirements=[],
        constraints=[],
        preferences=["professional tone"],
        missing_context=["recipient", "email purpose"],
    )
    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance("write a professional email", existing_analysis=analysis)
    assert "Email recipient" in enhanced or "recipient" in enhanced.lower()
    assert "- Email tone" not in enhanced
    assert "Email tone" not in enhanced


@pytest.mark.asyncio
async def test_consistency_f_preserves_legitimate_missing_context():
    """TEST F: Consistency cleanup does NOT remove legitimate missing context like recipient, email purpose, data storage."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "WRITING", "confidence": 0.95},
        "objective": "Draft a professional email",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["professional tone"],
        "missing_context": [
            "recipient",
            "email purpose",
            "data storage method",
            "user authentication requirements",
        ],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("write a professional email")
    assert "recipient" in result.missing_context
    assert "email purpose" in result.missing_context
    assert "data storage method" in result.missing_context
    assert "user authentication requirements" in result.missing_context


@pytest.mark.asyncio
async def test_enhancer_discipline_a_portfolio_no_candidate_tech_in_missing_context():
    """TEST A: 'build portfolio website' missing_context must not contain candidate technologies like React, Angular, Vue, etc."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a portfolio website",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "desired platform (e.g., React, Angular, Vue)",
            "styling framework (such as Tailwind)",
            "backend or cloud provider (e.g., Firebase, AWS, Next.js)",
        ],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build portfolio website")
    forbidden_techs = ["react", "angular", "vue", "next.js", "tailwind", "firebase", "aws"]
    for item in result.missing_context:
        for tech in forbidden_techs:
            assert tech not in item.lower()


@pytest.mark.asyncio
async def test_enhancer_discipline_b_missing_context_strips_parenthetical_examples():
    """TEST B: missing_context must not contain '(e.g.', 'for example', 'such as' for representative mocked analysis."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build an app",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "target platform (e.g., web, iOS, Android)",
            "data persistence (for example, relational database)",
            "styling preferences (such as modern dark theme)",
        ],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build an app")
    for item in result.missing_context:
        assert "(e.g." not in item.lower()
        assert "for example" not in item.lower()
        assert "such as" not in item.lower()


@pytest.mark.asyncio
async def test_enhancer_discipline_c_attendance_app_no_invented_faculty_or_teacher_role():
    """TEST C: 'make python attendance app for college' must NOT contain invented roles like college faculty member, teacher, student."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "ACT AS A COLLEGE FACULTY MEMBER.\n"
        "Create a Python attendance app for college.\n\n"
        "EXPLICIT REQUIREMENTS\n- Python\n\n"
        "UNSPECIFIED DETAILS\n- application interface type\n- data storage method\n\n"
        "OUTPUT EXPECTATIONS\nProvide the code."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a Python attendance app for college",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=["application interface type", "data storage method"],
    )
    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance("make python attendance app for college", existing_analysis=analysis)

    forbidden_roles = ["college faculty member", "faculty member", "teacher", "administrator", "student role"]
    for role in forbidden_roles:
        assert role not in enhanced.lower()


@pytest.mark.asyncio
async def test_enhancer_discipline_d_professional_email_preserves_professional_tone():
    """TEST D: 'write a professional email' enhanced output must preserve professional tone."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "ROLE & OBJECTIVE\nAct as a professional email writer. Write a professional email.\n\n"
        "OUTPUT EXPECTATIONS\nProvide the email."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.95),
        objective="Write a professional email",
        explicit_requirements=[],
        constraints=[],
        preferences=["professional tone"],
        missing_context=["recipient", "email purpose"],
    )
    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance("write a professional email", existing_analysis=analysis)
    assert "professional tone" in enhanced.lower() or "professional" in enhanced.lower()


@pytest.mark.asyncio
async def test_enhancer_discipline_e_professional_email_preserves_recipient_and_purpose():
    """TEST E: 'write a professional email' enhanced output must preserve recipient and email purpose under unspecified details."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "ROLE & OBJECTIVE\nAct as a professional email writer. Write a professional email.\n\n"
        "OUTPUT EXPECTATIONS\nProvide the email."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.95),
        objective="Write a professional email",
        explicit_requirements=[],
        constraints=[],
        preferences=["professional tone"],
        missing_context=["recipient", "email purpose"],
    )
    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance("write a professional email", existing_analysis=analysis)
    assert "recipient" in enhanced.lower()
    assert "email purpose" in enhanced.lower() or "purpose" in enhanced.lower()


@pytest.mark.asyncio
async def test_enhancer_discipline_f_explicit_role_preservation():
    """TEST F: When user explicitly asks for a role ('Act as a college faculty member...'), the role must be preserved."""
    user_prompt = "Act as a college faculty member and design an attendance workflow"
    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "ROLE & OBJECTIVE\n"
        "Act as a college faculty member and design an attendance workflow for university classes.\n\n"
        "OUTPUT EXPECTATIONS\nProvide the workflow."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.PLANNING, confidence=0.95),
        objective="Design an attendance workflow",
        explicit_requirements=["college faculty member"],
        constraints=[],
        preferences=[],
        missing_context=["class size"],
    )
    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance(user_prompt, existing_analysis=analysis)
    assert "college faculty member" in enhanced.lower()


# ==============================================================================
# PHASE 4 TESTS: CATEGORY-SPECIFIC ENHANCEMENT TEMPLATES & SELECTOR
# ==============================================================================

@pytest.mark.parametrize(
    "category,expected_class,expected_name",
    [
        (PromptCategory.CODING, CodingTemplate, "Coding"),
        (PromptCategory.DEBUGGING, DebuggingTemplate, "Debugging"),
        (PromptCategory.RESEARCH, ResearchTemplate, "Research"),
        (PromptCategory.EDUCATION, EducationTemplate, "Education"),
        (PromptCategory.WRITING, WritingTemplate, "Writing"),
        (PromptCategory.DATA_ANALYSIS, DataAnalysisTemplate, "Data Analysis"),
        (PromptCategory.PLANNING, PlanningTemplate, "Planning"),
        (PromptCategory.CREATIVE, CreativeTemplate, "Creative"),
        (PromptCategory.IMAGE_GENERATION, ImageGenerationTemplate, "Image Generation"),
        (PromptCategory.SUMMARIZATION, SummarizationTemplate, "Summarization"),
        (PromptCategory.GENERAL, GeneralTemplate, "General"),
    ],
)
def test_template_selector_all_categories(category, expected_class, expected_name):
    """TEST 1-11: Every locked category must map to its dedicated template instance."""
    template = get_template(category)
    assert isinstance(template, expected_class)
    assert template.name == expected_name
    assert template.category == category

    # Also verify string representations
    template_str = get_template(category.value)
    assert isinstance(template_str, expected_class)
    template_lower = get_template(category.value.lower())
    assert isinstance(template_lower, expected_class)


def test_template_selector_unknown_category_fallback():
    """TEST 12: Unknown or invalid categories must safely fall back to GeneralTemplate without raising."""
    invalid_inputs = ["INVALID_CATEGORY", "NONEXISTENT", "", None, 12345]
    for inp in invalid_inputs:
        template = get_template(inp)  # type: ignore
        assert isinstance(template, GeneralTemplate)
        assert template.category == PromptCategory.GENERAL


@pytest.mark.asyncio
async def test_category_behavior_education_explain_binary_search():
    """TEST A: 'explain binary search simply' uses EducationTemplate with simple, pedagogical guidance."""
    user_prompt = "explain binary search simply"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.EDUCATION, confidence=0.95),
        objective="Explain binary search simply",
        explicit_requirements=[],
        constraints=[],
        preferences=["simple explanation"],
        missing_context=[],
    )

    template = get_template(analysis.classification.category)
    assert isinstance(template, EducationTemplate)

    instructions = template.get_instructions(user_prompt, analysis)
    assert "Pedagogical Instruction" in instructions
    assert "simple explanation" in instructions
    # Must NOT mandate software architecture
    assert "TECHNICAL REQUIREMENTS" not in instructions
    assert "framework" not in instructions.lower()

    # Enhanced user message format check
    from app.prompts.enhancer import build_enhancer_user_message
    msg = build_enhancer_user_message(user_prompt, analysis, template=template)
    assert "EDUCATION" in msg
    assert "explain binary search simply" in msg


@pytest.mark.asyncio
async def test_category_behavior_debugging_fastapi_500():
    """TEST B: 'my FastAPI endpoint returns 500 fix it' uses DebuggingTemplate focusing on defect diagnosis."""
    user_prompt = "my FastAPI endpoint returns 500 fix it"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.DEBUGGING, confidence=0.95),
        objective="Fix FastAPI endpoint returning 500 error",
        explicit_requirements=["FastAPI"],
        constraints=[],
        preferences=[],
        missing_context=["error logs", "stack trace"],
    )

    template = get_template(analysis.classification.category)
    assert isinstance(template, DebuggingTemplate)

    instructions = template.get_instructions(user_prompt, analysis)
    assert "Defect Diagnosis" in instructions
    assert "OBSERVED BEHAVIOR" in instructions
    # Must NOT treat as building a new API
    assert "Never treat the request as building a brand new application" in instructions


@pytest.mark.asyncio
async def test_category_behavior_writing_professional_email():
    """TEST C: 'write a professional email' uses WritingTemplate preserving tone and omitting software sections."""
    user_prompt = "write a professional email"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.95),
        objective="Write a professional email",
        explicit_requirements=[],
        constraints=[],
        preferences=["professional tone"],
        missing_context=["recipient", "email purpose"],
    )

    template = get_template(analysis.classification.category)
    assert isinstance(template, WritingTemplate)

    instructions = template.get_instructions(user_prompt, analysis)
    assert "Textual Communication" in instructions
    assert "professional tone" in instructions
    # Must NOT produce software sections
    assert "Never output software or code sections" in instructions


@pytest.mark.asyncio
async def test_category_behavior_research_blockchain():
    """TEST D: 'research blockchain in sustainable finance' uses ResearchTemplate without coding sections."""
    user_prompt = "research blockchain in sustainable finance"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.RESEARCH, confidence=0.95),
        objective="Research blockchain applications in sustainable finance",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    template = get_template(analysis.classification.category)
    assert isinstance(template, ResearchTemplate)

    instructions = template.get_instructions(user_prompt, analysis)
    assert "Comprehensive Investigation" in instructions
    assert "RESEARCH QUESTION" in instructions
    # Must not output coding sections
    assert "Never output software development or coding headings" in instructions


@pytest.mark.asyncio
async def test_category_behavior_coding_react_portfolio():
    """TEST E: 'build React portfolio website' uses CodingTemplate preserving React and avoiding invented tech."""
    user_prompt = "build React portfolio website"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a React portfolio website",
        explicit_requirements=["React"],
        constraints=[],
        preferences=[],
        missing_context=["portfolio content requirements"],
    )

    template = get_template(analysis.classification.category)
    assert isinstance(template, CodingTemplate)

    instructions = template.get_instructions(user_prompt, analysis)
    assert "Software Implementation" in instructions
    assert "React" in instructions
    for anti in ["Next.js", "Tailwind", "Firebase"]:
        assert anti in "".join(template.anti_patterns)


@pytest.mark.asyncio
async def test_category_behavior_image_generation():
    """TEST F: 'create an image prompt for a futuristic Indian city' uses ImageGenerationTemplate without buzzword spam."""
    user_prompt = "create an image prompt for a futuristic Indian city"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.IMAGE_GENERATION, confidence=0.95),
        objective="Create an image prompt for a futuristic Indian city",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    template = get_template(analysis.classification.category)
    assert isinstance(template, ImageGenerationTemplate)

    instructions = template.get_instructions(user_prompt, analysis)
    assert "Visual Prompt Optimization" in instructions
    assert "LIGHTING & COLOR" in instructions
    # Must ban generic buzzwords
    assert "8K" in instructions or "8K" in "".join(template.anti_patterns)


@pytest.mark.asyncio
async def test_no_extra_llm_call_in_enhance_pipeline():
    """Verify that enhance pipeline makes exactly 1 analysis LLM call and 1 enhancement LLM call."""
    mock_analyzer = AsyncMock()
    analysis_obj = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build app",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    mock_analyzer.analyze.return_value = analysis_obj

    mock_client = AsyncMock()
    mock_client.chat.return_value = "Enhanced code prompt"

    enhancer = PromptEnhancer(client=mock_client, analyzer=mock_analyzer)
    returned_analysis, enhanced_text = await enhancer.enhance("make python app")

    # Analyzer called exactly once (1 LLM call)
    mock_analyzer.analyze.assert_called_once_with("make python app")
    # Chat called exactly once for enhancement (1 LLM call)
    mock_client.chat.assert_called_once()
    assert enhanced_text == "Enhanced code prompt"
    assert returned_analysis == analysis_obj


# --- Phase 4 Robustness & Recovery Regression Tests ---

@pytest.mark.asyncio
async def test_recovery_a_missing_objective_repaired():
    """Test A: Analyzer missing 'objective' on attempt 1 recovers on attempt 2."""
    mock_client = AsyncMock()
    # First response: missing required 'objective'
    attempt_1 = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.9},
        "explicit_requirements": ["Python"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    # Second response: fully valid PromptAnalysis
    attempt_2 = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build attendance app",
        "explicit_requirements": ["Python"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["application interface type"],
    })
    mock_client.chat.side_effect = [attempt_1, attempt_2]

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make python attendance app")

    assert mock_client.chat.call_count == 2
    assert result.objective == "Build attendance app"
    assert "Python" in result.explicit_requirements
    assert result.classification.category == PromptCategory.CODING


@pytest.mark.asyncio
async def test_recovery_b_malformed_json_repaired():
    """Test B: Analyzer malformed JSON on attempt 1 recovers on attempt 2."""
    mock_client = AsyncMock()
    attempt_1 = "This is definitely { not valid json at all"
    attempt_2 = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.94},
        "objective": "Build portfolio site",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["development platform or framework"],
    })
    mock_client.chat.side_effect = [attempt_1, attempt_2]

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build portfolio website")

    assert mock_client.chat.call_count == 2
    assert result.objective == "Build portfolio site"
    assert result.classification.category == PromptCategory.CODING


@pytest.mark.asyncio
async def test_recovery_c_both_attempts_invalid_controlled_error():
    """Test C: Both attempts invalid raises clean controlled backend error with clean HTTP 502."""
    mock_client = AsyncMock()
    attempt_1 = json.dumps({"classification": {"category": "CODING", "confidence": 0.9}})
    attempt_2 = json.dumps({"classification": {"category": "CODING", "confidence": 0.9}})
    mock_client.chat.side_effect = [attempt_1, attempt_2]

    analyzer = PromptAnalyzer(client=mock_client)
    with pytest.raises(OllamaResponseError) as exc_info:
        await analyzer.analyze("build app")

    # Error must not contain raw Python tracebacks
    assert "Traceback" not in str(exc_info.value)

    # Through FastAPI route, frontend receives clean human-readable error
    with patch("app.api.routes.prompt_analyzer.analyze", side_effect=OllamaResponseError(
        "HumanIntent couldn't produce a valid structured analysis for this request (does not match the required schema): 1 validation error"
    )):
        response = client.post("/api/analyze", json={"prompt": "build app"})
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "HumanIntent couldn't produce a valid structured analysis for this request. Please try again or rephrase the request."
        assert "Traceback" not in data["detail"]
        assert "validation error" not in data["detail"].lower()


@pytest.mark.asyncio
async def test_self_correction_d_quiz_over_calculator():
    """Test D: 'maybe make a calculator, no make it a quiz' -> quiz is final intent, calculator discarded."""
    mock_client = AsyncMock()
    # Even if LLM naively returned calculator in explicit requirements:
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build an interactive quiz",
        "explicit_requirements": ["calculator", "quiz"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("maybe make a calculator, no make it a quiz")

    assert result.classification.category == PromptCategory.CODING
    assert "quiz" in result.objective.lower()
    assert not any("calculator" in req.lower() for req in result.explicit_requirements)


@pytest.mark.asyncio
async def test_self_correction_e_technology_react_to_vue():
    """Test E: 'use React, actually use Vue' -> Vue is preserved, React is not treated as active requirement."""
    mock_client = AsyncMock()
    # Model naively outputs both or none:
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build dashboard",
        "explicit_requirements": ["React", "Vue"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("use React, actually use Vue")

    assert "Vue" in result.explicit_requirements
    assert "React" not in result.explicit_requirements


@pytest.mark.asyncio
async def test_self_correction_f_preference_formal_to_friendly():
    """Test F: 'make it formal, actually friendly' -> friendly is preserved as final preference."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "WRITING", "confidence": 0.95},
        "objective": "Write an announcement email",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["formal tone", "friendly tone"],
        "missing_context": ["recipient", "email purpose"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("make it formal, actually friendly")

    assert any("friendly" in p.lower() for p in result.preferences)
    assert not any("formal" in p.lower() for p in result.preferences)


@pytest.mark.asyncio
async def test_longer_messy_conversational_prompt_mocked_pipeline():
    """Test G: Longer messy conversational prompt works through pipeline without crashing."""
    raw_prompt = (
        "so basically i've been thinking about this idea for my site for a while now, "
        "like you know how people want to learn coding? maybe like a tutorial? wait no, "
        "i want an interactive python code playground where users can type code and see output, "
        "or whatever make it look modern and sleek please"
    )
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Create an interactive Python code playground",
        "explicit_requirements": ["Python", "interactive code execution"],
        "constraints": [],
        "preferences": ["modern and sleek style"],
        "missing_context": ["application interface type"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(raw_prompt)

    assert result.classification.category == PromptCategory.CODING
    assert "Python" in result.explicit_requirements
    assert not any("tutorial" in req.lower() for req in result.explicit_requirements)


def test_input_length_10000_limit_preserved():
    """Test H: Existing 10,000-character validation limit remains strictly active."""
    valid_prompt = "a" * 10000
    invalid_prompt = "a" * 10001

    with patch("app.api.routes.prompt_analyzer.analyze", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = PromptAnalysis(
            classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
            objective="Test objective",
            explicit_requirements=[],
            constraints=[],
            preferences=[],
            missing_context=[],
        )
        res_valid = client.post("/api/analyze", json={"prompt": valid_prompt})
        assert res_valid.status_code == 200

    res_invalid = client.post("/api/analyze", json={"prompt": invalid_prompt})
    assert res_invalid.status_code == 422


@pytest.mark.asyncio
async def test_exact_failure_shoe_quiz_regression():
    """Test exact failure prompt regression semantics."""
    exact_prompt = (
        "so i need a thing for my website like a page but not a normal page something where people click "
        "and it shows them stuff wait maybe like a calculator? no like a quiz but with results at the end "
        "that tell them what shoes to buy based on their vibe or whatever make it sound cool"
    )

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.92},
        "objective": "Create an interactive website quiz that recommends shoes based on user results",
        "explicit_requirements": ["interactive quiz", "shoe recommendations based on results"],
        "constraints": [],
        "preferences": ["cool and engaging style"],
        "missing_context": ["shoe catalogue or recommendation data"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(exact_prompt)

    # 1. Classification is CODING
    assert result.classification.category == PromptCategory.CODING
    # 2. Objective is interactive quiz / shoe recommendations
    assert any(w in result.objective.lower() for w in ["quiz", "shoe", "recommend"])
    # 3. Discarded calculator thought must NOT become an explicit requirement
    assert not any("calculator" in req.lower() for req in result.explicit_requirements)
    # 4. Preference preserves cool / engaging style
    assert any("cool" in p.lower() for p in result.preferences)


# --- Phase 4 Deliverable Classification & Anti-Invention Tests ---

@pytest.mark.asyncio
async def test_anti_invention_a_shoe_quiz_no_hallucinations():
    """Test A: Exact shoe quiz enhancement preserves intent and never invents price, brands, sizes, demographics, responsive, privacy, security."""
    exact_prompt = (
        "so i need a thing for my website like a page but not a normal page something where people click "
        "and it shows them stuff wait maybe like a calculator? no like a quiz but with results at the end "
        "that tell them what shoes to buy based on their vibe or whatever make it sound cool"
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create an interactive website quiz that recommends shoes based on user results",
        explicit_requirements=["interactive quiz", "shoe recommendations based on results"],
        constraints=[],
        preferences=["cool and engaging style"],
        missing_context=["shoe recommendation data source"],
    )

    mock_client = AsyncMock()
    # Mock model that attempted to inject generic unrequested features
    mock_client.chat.return_value = (
        "OBJECTIVE:\n"
        "Create an interactive website quiz that recommends shoes based on user results.\n\n"
        "CORE REQUIREMENTS:\n"
        "- Interactive quiz experience\n"
        "- Show shoe recommendations at the end\n"
        "- Display the recommended shoes with a brief description and price information\n\n"
        "PREFERENCES:\n"
        "- Cool and engaging style\n\n"
        "UNSPECIFIED DETAILS:\n"
        "- User demographics (e.g. age, gender, location)\n"
        "- Shoe options (e.g. styles, brands, sizes)\n"
        "- shoe recommendation data source\n\n"
        "OUTPUT EXPECTATIONS:\n"
        "- Implement the quiz and shoe recommendation system in a modern web application\n"
        "- Use a responsive design to ensure the application is accessible on various devices\n"
        "- Ensure the application is secure and follows best practices for user data and privacy"
    )

    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance(exact_prompt, existing_analysis=analysis)

    enhanced_lower = enhanced.lower()

    # Must preserve:
    assert "quiz" in enhanced_lower
    assert "shoe" in enhanced_lower
    assert "cool" in enhanced_lower or "engaging" in enhanced_lower

    # Must NOT introduce unrequested features:
    assert "price" not in enhanced_lower
    assert "demographic" not in enhanced_lower
    assert "age" not in enhanced_lower
    assert "gender" not in enhanced_lower
    assert "location" not in enhanced_lower
    assert "brand" not in enhanced_lower
    assert "sizes" not in enhanced_lower
    assert "responsive" not in enhanced_lower
    assert "privacy" not in enhanced_lower
    assert "security" not in enhanced_lower
    assert "authentication" not in enhanced_lower


@pytest.mark.asyncio
async def test_classification_b_quiz_page_is_coding():
    """Test B: 'create a quiz page on my website that recommends shoes' -> CODING."""
    mock_client = AsyncMock()
    # Even if raw model returned CREATIVE:
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CREATIVE", "confidence": 0.9},
        "objective": "Create a quiz page on my website that recommends shoes",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("create a quiz page on my website that recommends shoes")
    assert result.classification.category == PromptCategory.CODING


@pytest.mark.asyncio
async def test_classification_c_brainstorm_is_creative():
    """Test C: 'brainstorm fun ideas for a shoe personality quiz' -> CREATIVE."""
    mock_client = AsyncMock()
    # Even if raw model returned CODING:
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.9},
        "objective": "Brainstorm fun ideas for a shoe personality quiz",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("brainstorm fun ideas for a shoe personality quiz")
    assert result.classification.category == PromptCategory.CREATIVE


@pytest.mark.asyncio
async def test_classification_d_curriculum_is_education_not_coding():
    """Test D: 'create an 8-week curriculum to teach HTML CSS and JavaScript' -> EDUCATION (not CODING)."""
    mock_client = AsyncMock()
    # Even if raw model saw HTML/CSS/JavaScript and returned CODING:
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.9},
        "objective": "Create an 8-week curriculum to teach HTML CSS and JavaScript",
        "explicit_requirements": ["HTML", "CSS", "JavaScript"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("create an 8-week curriculum to teach HTML CSS and JavaScript")
    assert result.classification.category == PromptCategory.EDUCATION
    assert result.classification.category != PromptCategory.CODING


@pytest.mark.asyncio
async def test_classification_e_portfolio_is_coding():
    """Test E: 'build an HTML CSS JavaScript portfolio' -> CODING."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build an HTML CSS JavaScript portfolio",
        "explicit_requirements": ["HTML", "CSS", "JavaScript"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })
    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze("build an HTML CSS JavaScript portfolio")
    assert result.classification.category == PromptCategory.CODING


@pytest.mark.asyncio
async def test_user_requested_feature_preservation_f():
    """Test F: When user explicitly requests 'responsive' and 'prices', they MUST be preserved."""
    prompt = "create a responsive quiz website with shoe prices"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a responsive quiz website with shoe prices",
        explicit_requirements=["responsive design", "shoe prices"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "OBJECTIVE:\n"
        "Create a responsive quiz website with shoe prices.\n\n"
        "EXPLICIT REQUIREMENTS:\n"
        "- Responsive design\n"
        "- Shoe prices\n\n"
        "OUTPUT EXPECTATIONS:\n"
        "- Use a responsive design layout\n"
        "- Display shoe price information"
    )

    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance(prompt, existing_analysis=analysis)

    enhanced_lower = enhanced.lower()
    # Explicitly requested features MUST be kept:
    assert "responsive" in enhanced_lower
    assert "price" in enhanced_lower


# --- Phase 4 Cross-Pipeline Consistency Tests (Tasks A-G) ---

@pytest.mark.asyncio
async def test_cross_pipeline_a_shoe_quiz_full_api_consistency():
    """Test A: Full /api/enhance consistency for the shoe quiz prompt preserves quiz, results, and shoe recommendations."""
    prompt = (
        "so i need a thing for my website like a page but not a normal page something where people click "
        "and it shows them stuff wait maybe like a calculator? no like a quiz but with results at the end "
        "that tell them what shoes to buy based on their vibe or whatever make it sound cool"
    )

    mock_analysis_json = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Create a quiz page on my website that recommends shoes based on user vibe",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["cool and engaging style"],
        "missing_context": ["shoe recommendation data source"],
    })

    with patch("app.core.analyzer.ollama_client.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = [
            mock_analysis_json,  # analyzer LLM call
            "Act as an experienced software developer.\nCreate a quiz page on my website that recommends shoes.\n\nCORE REQUIREMENTS:\n- Quiz\n- Results\n- Shoe recommendations\n\nOUTPUT EXPECTATIONS:\nImplement the quiz cleanly.",  # enhancer LLM call
            json.dumps({"is_valid": True, "issues": []}),  # validator LLM call
        ]

        response = client.post("/api/enhance", json={"prompt": prompt})
        assert response.status_code == 200
        data = response.json()

        analysis = data["analysis"]
        reqs_lower = [r.lower() for r in analysis["explicit_requirements"]]

        # Assert explicit_requirements preserves all three
        assert any("quiz" in r for r in reqs_lower)
        assert any("result" in r for r in reqs_lower)
        assert any("shoe" in r and "recommend" in r for r in reqs_lower)

        # Calculator must NOT be present
        assert not any("calculator" in r for r in reqs_lower)

        # Analysis returned in EnhanceResponse matches structure expected by UnderstoodIntent
        assert "classification" in analysis
        assert "objective" in analysis
        assert "constraints" in analysis
        assert "preferences" in analysis
        assert "missing_context" in analysis


@pytest.mark.asyncio
async def test_cross_pipeline_b_website_context_no_interface_missing():
    """Test B: 'create a quiz page on my website' must NOT report interface type as missing."""
    prompt = "create a quiz page on my website that recommends shoes"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Create a quiz page on my website that recommends shoes",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["quiz interface type", "data storage method"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    missing_lower = [m.lower() for m in result.missing_context]
    assert not any("interface" in m for m in missing_lower)
    assert not any("quiz interface type" in m for m in missing_lower)


@pytest.mark.asyncio
async def test_cross_pipeline_c_creative_brainstorming_no_narrative_tropes():
    """Test C: 'brainstorm fun ideas for a shoe personality quiz' must NOT invent vivid imagery or narrative voice."""
    prompt = "brainstorm fun ideas for a shoe personality quiz"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CREATIVE, confidence=0.95),
        objective="Brainstorm fun ideas for a shoe personality quiz",
        explicit_requirements=[],
        constraints=[],
        preferences=["fun tone"],
        missing_context=[],
    )

    mock_client = AsyncMock()
    # Mock model that naively regurgitated storytelling tropes
    mock_client.chat.return_value = (
        "OBJECTIVE:\n"
        "Brainstorm fun ideas for a shoe personality quiz.\n\n"
        "PREFERENCES:\n"
        "- Fun tone\n\n"
        "Start generating creative concepts that bring the user's idea to life with vivid imagery and compelling narrative voice, using an evocative and engaging tone."
    )

    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance(prompt, existing_analysis=analysis)

    enhanced_lower = enhanced.lower()
    assert "vivid imagery" not in enhanced_lower
    assert "narrative voice" not in enhanced_lower
    assert "evocative" not in enhanced_lower


@pytest.mark.asyncio
async def test_cross_pipeline_d_education_explicit_preservation():
    """Test D: 'create an 8-week curriculum to teach HTML CSS and JavaScript' preserves 8-week, HTML, CSS, JavaScript."""
    prompt = "create an 8-week curriculum to teach HTML CSS and JavaScript"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "EDUCATION", "confidence": 0.95},
        "objective": "Create an 8-week curriculum to teach HTML, CSS, and JavaScript",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["learners' prior knowledge level"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    # Classification is EDUCATION
    assert result.classification.category == PromptCategory.EDUCATION

    # Technologies preserved in explicit_requirements
    reqs = result.explicit_requirements
    assert "HTML" in reqs
    assert "CSS" in reqs
    assert "JavaScript" in reqs

    # 8-week duration preserved in constraints or explicit_requirements
    all_bounds = [c.lower() for c in result.constraints] + [r.lower() for r in result.explicit_requirements]
    assert any("8-week" in b or "8 week" in b for b in all_bounds)


@pytest.mark.asyncio
async def test_cross_pipeline_e_portfolio_neutrality():
    """Test E: 'build an HTML CSS JavaScript portfolio' missing_context must NOT contain candidate sections/features."""
    prompt = "build an HTML CSS JavaScript portfolio"
    mock_client = AsyncMock()
    # Model naively generated candidate sections as missing context:
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build an HTML, CSS, and JavaScript portfolio",
        "explicit_requirements": ["HTML", "CSS", "JavaScript"],
        "constraints": [],
        "preferences": [],
        "missing_context": [
            "project showcase",
            "resume",
            "contact page",
            "personal branding",
            "digital footprint",
        ],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    missing_lower = [m.lower() for m in result.missing_context]
    forbidden_features = ["project showcase", "resume", "contact page", "personal branding", "digital footprint"]
    for feat in forbidden_features:
        assert not any(feat in m for m in missing_lower), f"Forbidden candidate feature '{feat}' found in missing_context: {result.missing_context}"


@pytest.mark.asyncio
async def test_cross_pipeline_f_style_discipline_cool_not_friendly():
    """Test F: 'make it sound cool' must not automatically produce 'friendly tone'."""
    prompt = "create a shoe quiz and make it sound cool"
    mock_client = AsyncMock()
    # Model hallucinated friendly tone:
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Create a shoe quiz",
        "explicit_requirements": ["shoe quiz"],
        "constraints": [],
        "preferences": ["cool tone", "friendly tone"],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    pref_lower = [p.lower() for p in result.preferences]
    assert any("cool" in p for p in pref_lower)
    assert not any("friendly" in p for p in pref_lower)


@pytest.mark.asyncio
async def test_cross_pipeline_g_empty_section_removal():
    """Test G: Enhanced output must not contain a heading with no meaningful content."""
    prompt = "make python script"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Make a Python script",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    mock_client = AsyncMock()
    # Output ending with empty OUTPUT EXPECTATIONS heading
    mock_client.chat.return_value = (
        "Act as an experienced software developer.\n"
        "Make a Python script.\n\n"
        "CORE REQUIREMENTS:\n"
        "- Python\n\n"
        "OUTPUT EXPECTATIONS:\n"
    )

    enhancer = PromptEnhancer(client=mock_client)
    _, enhanced = await enhancer.enhance(prompt, existing_analysis=analysis)

    # Empty section heading must be stripped
    assert not enhanced.strip().endswith("OUTPUT EXPECTATIONS:")
    assert "OUTPUT EXPECTATIONS:" not in enhanced


# ==============================================================================
# GENERALIZATION HARDENING: METAMORPHIC TESTS (Task 17)
# ==============================================================================

@pytest.mark.asyncio
async def test_metamorphic_quantity_and_referent_deliverables():
    """Metamorphic 1: 'give me 4 ideas and expand the second' vs 'come up with four ideas, then develop idea #2'."""
    prompt_a = "give me 4 ideas and expand the second"
    prompt_b = "come up with four ideas, then develop idea #2"

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CREATIVE", "confidence": 0.90},
        "objective": "Brainstorm ideas and develop one",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res_a = await analyzer.analyze(prompt_a)
    res_b = await analyzer.analyze(prompt_b)

    reqs_a = [r.lower() for r in res_a.explicit_requirements]
    reqs_b = [r.lower() for r in res_b.explicit_requirements]

    # Both must preserve the quantity (4 ideas)
    assert any("4" in r and "idea" in r for r in reqs_a)
    assert any("4" in r and "idea" in r for r in reqs_b)

    # Both must preserve the drafting/expansion of the second item
    assert any("draft" in r or "expand" in r for r in reqs_a)
    assert any("draft" in r or "develop" in r for r in reqs_b)


@pytest.mark.asyncio
async def test_metamorphic_self_correction_syntax():
    """Metamorphic 2: 'build React app, actually use Vue' vs 'start with React — scratch that, make it Vue'."""
    prompt_a = "build React app, actually use Vue"
    prompt_b = "start with React — scratch that, make it Vue"

    mock_client = AsyncMock()
    # Mock LLM that mistakenly kept both React and Vue
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build application",
        "explicit_requirements": ["React", "Vue"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res_a = await analyzer.analyze(prompt_a)
    res_b = await analyzer.analyze(prompt_b)

    # In both cases, Vue is the final choice; React is superseded
    assert "Vue" in res_a.explicit_requirements
    assert "React" not in res_a.explicit_requirements

    assert "Vue" in res_b.explicit_requirements
    assert "React" not in res_b.explicit_requirements


@pytest.mark.asyncio
async def test_metamorphic_style_synonyms():
    """Metamorphic 3: 'keep it concise' vs 'make it brief' both map to equivalent concise style."""
    prompt_a = "write a summary of the quarterly report, keep it concise"
    prompt_b = "write a summary of the quarterly report, make it brief"

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "SUMMARIZATION", "confidence": 0.95},
        "objective": "Summarize quarterly report",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res_a = await analyzer.analyze(prompt_a)
    res_b = await analyzer.analyze(prompt_b)

    assert any("concise" in p.lower() for p in res_a.preferences)
    assert any("concise" in p.lower() for p in res_b.preferences)


@pytest.mark.asyncio
async def test_metamorphic_web_modality_phrasings():
    """Metamorphic 4: 'make a webpage' vs 'create a page for my website' both eliminate interface ambiguity."""
    prompt_a = "make a webpage for tracking personal expenses"
    prompt_b = "create a page for my website for tracking personal expenses"

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Track personal expenses",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["application interface type", "data storage method"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res_a = await analyzer.analyze(prompt_a)
    res_b = await analyzer.analyze(prompt_b)

    for res in [res_a, res_b]:
        missing_lower = [m.lower() for m in res.missing_context]
        assert not any("interface" in m for m in missing_lower)
        assert any("storage" in m for m in missing_lower)


# ==============================================================================
# GENERALIZATION HARDENING: CROSS-DOMAIN GENERALIZATION TESTS (Task 18)
# ==============================================================================

@pytest.mark.asyncio
async def test_cross_domain_1_science_fair_ideas_selection_explanation():
    """Domain 1: Science fair ideas with selection and explanation deliverables."""
    prompt = "give me 4 science fair ideas, choose the strongest one, and explain why"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CREATIVE", "confidence": 0.92},
        "objective": "Provide science fair ideas with recommendation and rationale",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["grade level or target age group"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    reqs_lower = [r.lower() for r in result.explicit_requirements]
    # Quantity preserved
    assert any("4" in r and "idea" in r for r in reqs_lower)
    # Selection action preserved
    assert any("select" in r or "choose" in r for r in reqs_lower)
    # Explanation action preserved
    assert any("explain" in r for r in reqs_lower)


@pytest.mark.asyncio
async def test_cross_domain_2_newsletter_subject_lines_referent_and_constraint_targeting():
    """Domain 2: Newsletter subject lines with referent resolution and targeted length constraint."""
    prompt = "write 3 newsletter subject lines, draft the second one, and keep the draft under 80 words"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "WRITING", "confidence": 0.95},
        "objective": "Create newsletter subject lines and draft selected option",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["newsletter topic", "target audience"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    assert result.classification.category == PromptCategory.WRITING
    reqs_lower = [r.lower() for r in result.explicit_requirements]
    assert any("3" in r and "subject line" in r for r in reqs_lower)
    assert any("draft" in r for r in reqs_lower)

    constraints_lower = [c.lower() for c in result.constraints]
    assert any("80 words" in c for c in constraints_lower)


@pytest.mark.asyncio
async def test_cross_domain_3_photosynthesis_compound_education():
    """Domain 3: Educational instruction with simple explanation preference and practice questions."""
    prompt = "explain photosynthesis simply, give an example, then create 4 practice questions"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "EDUCATION", "confidence": 0.95},
        "objective": "Explain photosynthesis and generate practice questions",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["target education level"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    assert result.classification.category == PromptCategory.EDUCATION
    assert any("simple" in p.lower() for p in result.preferences)
    reqs_lower = [r.lower() for r in result.explicit_requirements]
    assert any("4" in r and "question" in r for r in reqs_lower)


@pytest.mark.asyncio
async def test_cross_domain_4_battery_recycling_compound_research_table():
    """Domain 4: Research investigation comparing approaches with table output format."""
    prompt = "research battery recycling methods, compare the main approaches, and summarize them in a table"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "RESEARCH", "confidence": 0.90},
        "objective": "Research and compare battery recycling methods",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["battery chemistry types"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    assert result.classification.category == PromptCategory.RESEARCH
    reqs_lower = [r.lower() for r in result.explicit_requirements]
    assert any("compare" in r for r in reqs_lower)
    assert any("table" in r for r in reqs_lower)


@pytest.mark.asyncio
async def test_cross_domain_5_vue_landing_page_contact_form_negative_constraint():
    """Domain 5: Vue landing page with contact form deliverable and negative framework constraint."""
    prompt = "build a Vue landing page with a contact form but don't use Bootstrap"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a landing page",
        "explicit_requirements": ["Vue"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["content requirements"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    assert result.classification.category == PromptCategory.CODING
    assert "Vue" in result.explicit_requirements
    reqs_lower = [r.lower() for r in result.explicit_requirements]
    assert any("contact form" in r for r in reqs_lower)
    assert any("landing page" in r for r in reqs_lower)

    constraints_lower = [c.lower() for c in result.constraints]
    assert any("bootstrap" in c for c in constraints_lower)


@pytest.mark.asyncio
async def test_cross_domain_6_study_schedule_planning_priority_duration():
    """Domain 6: 5-day study schedule with priority and duration constraint."""
    prompt = "plan a 5-day study schedule, prioritize calculus, and keep weekends lighter"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "PLANNING", "confidence": 0.95},
        "objective": "Create a study schedule",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["daily hours available"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    assert result.classification.category == PromptCategory.PLANNING
    constraints_lower = [c.lower() for c in result.constraints]
    assert any("5-day" in c for c in constraints_lower)


# ==============================================================================
# PHASE 5: AMBIGUITY + HIGH-IMPACT CLARIFICATION TESTS
# ==============================================================================

# --- Test Group 1: Clear Requests Across Categories (Proceed Immediately) ---

@pytest.mark.parametrize(
    "clear_prompt,category",
    [
        ("build a Python CLI tool to parse markdown documents", PromptCategory.CODING),
        ("explain recursion simply using a visual example", PromptCategory.EDUCATION),
        ("write a professional email to my engineering team regarding tomorrow's sprint demo", PromptCategory.WRITING),
        ("research comparative methods for carbon capture and summarize in a table", PromptCategory.RESEARCH),
        ("create a 4-week study schedule for GRE exam preparation", PromptCategory.PLANNING),
    ],
)
@pytest.mark.asyncio
async def test_phase5_group1_clear_requests_proceed_without_clarification(clear_prompt, category):
    """Group 1: Sufficiently specified prompts across categories must have needs_clarification=False."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": category.value, "confidence": 0.95},
        "objective": f"Accomplish task for {clear_prompt}",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["deployment environment"],
        "ambiguity": {
            "score": 15,
            "level": "LOW",
            "needs_clarification": False,
            "reasons": [],
            "questions": [],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(clear_prompt)

    assert result.ambiguity.needs_clarification is False
    assert result.ambiguity.score < 30
    assert result.ambiguity.level == AmbiguityLevel.LOW
    assert len(result.ambiguity.questions) == 0


# --- Test Group 2: Genuinely Ambiguous Requests (Trigger Clarification) ---

@pytest.mark.parametrize(
    "vague_prompt",
    [
        "build me a system",
        "make an app",
        "write an email",
        "analyze data",
        "help me",
    ],
)
@pytest.mark.asyncio
async def test_phase5_group2_genuinely_ambiguous_requests_trigger_clarification(vague_prompt):
    """Group 2: Ultra-vague prompts lacking deliverable/domain must trigger high-impact questions and 0 enhancer calls."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "GENERAL", "confidence": 0.4},
        "objective": "Unknown task",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["target purpose", "scope"],
        "ambiguity": {
            "score": 85,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Unclear primary objective and platform"],
            "questions": [
                {
                    "id": "q1",
                    "question": "What is the primary purpose and deliverable of this task?",
                    "reason": "Essential to determine task scope.",
                    "target_dimension": "primary purpose",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(vague_prompt)

    assert result.ambiguity.needs_clarification is True
    assert result.ambiguity.score >= 65
    assert result.ambiguity.level == AmbiguityLevel.HIGH
    assert 1 <= len(result.ambiguity.questions) <= 3


@pytest.mark.asyncio
async def test_phase5_group2_ambiguous_request_enhancer_calls_zero_llm_for_enhancement():
    """Group 2: For an ambiguous request, enhancer must not generate an enhanced prompt (0 enhancer LLM calls)."""
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.5),
        objective="Build a system",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["system domain"],
        ambiguity=AmbiguityAssessment(
            score=85,
            level=AmbiguityLevel.HIGH,
            needs_clarification=True,
            reasons=["Critical missing platform and domain"],
            questions=[
                ClarificationQuestion(
                    id="q1",
                    question="What platform should this system run on?",
                    reason="Platform determines architecture.",
                    target_dimension="platform",
                )
            ],
        ),
    )

    mock_client = AsyncMock()
    enhancer = PromptEnhancer(client=mock_client, analyzer=mock_analyzer)

    analysis, enhanced_text = await enhancer.enhance("build me a system")

    # Analyzer was called, but enhancer LLM was NOT called
    mock_analyzer.analyze.assert_called_once()
    mock_client.chat.assert_not_called()
    assert enhanced_text == ""
    assert analysis.ambiguity.needs_clarification is True


def test_phase5_group2_discriminated_api_response():
    """Group 2: /api/enhance returns status='needs_clarification' for ambiguous request."""
    mock_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.5),
        objective="Build a system",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["system domain"],
        ambiguity=AmbiguityAssessment(
            score=85,
            level=AmbiguityLevel.HIGH,
            needs_clarification=True,
            reasons=["Unclear system purpose"],
            questions=[
                ClarificationQuestion(
                    id="q1",
                    question="What is the primary purpose and platform of this system?",
                    reason="Fundamental to architecture.",
                    target_dimension="platform and purpose",
                )
            ],
        ),
    )

    with patch("app.api.routes.prompt_enhancer.enhance", new_callable=AsyncMock) as mock_enhance:
        mock_enhance.return_value = (mock_analysis, "")

        response = client.post("/api/enhance", json={"prompt": "build me a system"})
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "needs_clarification"
        assert "questions" in data
        assert len(data["questions"]) == 1
        assert data["questions"][0]["id"] == "q1"
        assert "enhanced_prompt" not in data


# --- Test Group 3: Missing Context != Required Clarification ---

@pytest.mark.asyncio
async def test_phase5_group3_missing_context_does_not_force_clarification():
    """Group 3: Non-empty missing_context must NOT automatically trigger needs_clarification."""
    prompt = "explain recursion simply"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "EDUCATION", "confidence": 0.95},
        "objective": "Explain recursion simply",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["simple explanation"],
        "missing_context": ["target audience grade level", "preferred programming language context"],
        "ambiguity": {
            "score": 20,
            "level": "LOW",
            "needs_clarification": False,
            "reasons": [],
            "questions": [],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    # Missing context exists
    assert len(result.missing_context) == 2
    # But clarification is NOT required
    assert result.ambiguity.needs_clarification is False
    assert result.ambiguity.level == AmbiguityLevel.LOW
    assert len(result.ambiguity.questions) == 0


# --- Test Group 4: Answers Resolve Ambiguity Without Loops ---

@pytest.mark.asyncio
async def test_phase5_group4_clarification_answers_resolve_ambiguity():
    """Group 4: Supplying answers incorporates context and reduces ambiguity, avoiding clarification loops."""
    original_prompt = "build me a system"
    answers = [
        ClarificationAnswer(
            question_id="q1",
            question="What is the primary purpose and target platform of the system?",
            answer="A web application for tracking warehouse inventory in real time",
        )
    ]

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.92},
        "objective": "Build a web application for tracking warehouse inventory in real time",
        "explicit_requirements": ["web application", "warehouse inventory tracking"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["data storage method"],
        "ambiguity": {
            "score": 15,
            "level": "LOW",
            "needs_clarification": False,
            "reasons": [],
            "questions": [],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(original_prompt, clarification_answers=answers)

    assert result.ambiguity.needs_clarification is False
    assert result.ambiguity.score < 30
    assert result.ambiguity.level == AmbiguityLevel.LOW
    assert len(result.ambiguity.questions) == 0
    # Answers incorporated
    assert any("warehouse inventory" in r.lower() for r in result.explicit_requirements)


# --- Test Group 5: Reasonable Assumptions Option ---

@pytest.mark.asyncio
async def test_phase5_group5_reasonable_assumptions_bypasses_clarification():
    """Group 5: allow_reasonable_assumptions=True allows enhancement without further questions."""
    original_prompt = "build me a system"

    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.7),
        objective="Build a general system",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["system domain"],
        ambiguity=AmbiguityAssessment(
            score=70,
            level=AmbiguityLevel.HIGH,
            needs_clarification=False,  # forced False by reasonable assumptions
            reasons=[],
            questions=[],
        ),
    )

    mock_client = AsyncMock()
    mock_client.chat.return_value = (
        "OBJECTIVE:\nDesign a system architecture.\n\n"
        "UNSPECIFIED DETAILS:\nIf implementation choices are required, assume standard defaults."
    )

    enhancer = PromptEnhancer(client=mock_client, analyzer=mock_analyzer)
    analysis, enhanced_text = await enhancer.enhance(
        original_prompt,
        allow_reasonable_assumptions=True,
    )

    # Enhancer was called because clarification was bypassed
    mock_client.chat.assert_called_once()
    assert enhanced_text != ""
    # Verify assumptions directive was sent
    call_args = mock_client.chat.call_args[1]
    user_msg = call_args["messages"][1]["content"]
    assert "REASONABLE ASSUMPTIONS DIRECTIVE" in user_msg


# --- Test Group 6: Question Limit Defensively Capped at 3 ---

def test_phase5_group6_questions_defensively_capped_at_three():
    """Group 6: Model emitting > 3 questions has them capped at maximum 3 with deterministic IDs."""
    data = {
        "score": 80,
        "level": "HIGH",
        "needs_clarification": True,
        "reasons": ["Unclear scope"],
        "questions": [
            {"id": "custom_1", "question": "Question 1?", "reason": "r1", "target_dimension": "d1"},
            {"id": "custom_2", "question": "Question 2?", "reason": "r2", "target_dimension": "d2"},
            {"id": "custom_3", "question": "Question 3?", "reason": "r3", "target_dimension": "d3"},
            {"id": "custom_4", "question": "Question 4?", "reason": "r4", "target_dimension": "d4"},
            {"id": "custom_5", "question": "Question 5?", "reason": "r5", "target_dimension": "d5"},
        ],
    }

    assessment = AmbiguityAssessment.model_validate(data)
    assert len(assessment.questions) == 3
    assert [q.id for q in assessment.questions] == ["q1", "q2", "q3"]


# --- Test Group 7: Score Validation and Normalization ---

def test_phase5_group7_score_clamping_and_normalization():
    """Group 7: Scores outside 0-100 are clamped and levels normalized."""
    # Negative score
    low_data = AmbiguityAssessment.model_validate({"score": -10, "needs_clarification": True})
    assert low_data.score == 0
    assert low_data.level == AmbiguityLevel.LOW
    assert low_data.needs_clarification is False
    assert low_data.questions == []

    # Score > 100
    high_data = AmbiguityAssessment.model_validate({"score": 150})
    assert high_data.score == 100
    assert high_data.level == AmbiguityLevel.HIGH

    # LOW ambiguity forces needs_clarification=False
    forced_low = AmbiguityAssessment.model_validate({
        "score": 15,
        "needs_clarification": True,
        "questions": [{"id": "q1", "question": "Test?", "reason": "r", "target_dimension": "d"}],
    })
    assert forced_low.needs_clarification is False
    assert forced_low.questions == []


# --- Test Group 8: Exact LLM Call Count Invariants ---

@pytest.mark.asyncio
async def test_phase5_group8_clear_path_exact_call_counts():
    """Group 8: Clear request requires exactly 1 analyzer call + 1 enhancer call (2 LLM calls total)."""
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a calculator",
        explicit_requirements=["calculator"],
        constraints=[],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(
            score=10,
            level=AmbiguityLevel.LOW,
            needs_clarification=False,
            reasons=[],
            questions=[],
        ),
    )

    mock_client = AsyncMock()
    mock_client.chat.return_value = "Enhanced prompt for calculator"

    enhancer = PromptEnhancer(client=mock_client, analyzer=mock_analyzer)
    analysis, enhanced = await enhancer.enhance("build a web calculator using Vue")

    # Invariant: 1 analyzer call + 1 enhancer call
    mock_analyzer.analyze.assert_called_once()
    mock_client.chat.assert_called_once()
    assert enhanced != ""


@pytest.mark.asyncio
async def test_phase5_group8_clarification_path_exact_call_counts():
    """Group 8: Initial ambiguous request requires exactly 1 analyzer call + 0 enhancer calls (1 LLM call total)."""
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.GENERAL, confidence=0.4),
        objective="Help me",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(
            score=90,
            level=AmbiguityLevel.HIGH,
            needs_clarification=True,
            reasons=["Unclear task"],
            questions=[
                ClarificationQuestion(
                    id="q1",
                    question="What specific task or problem would you like assistance with?",
                    reason="Needed to understand request.",
                    target_dimension="primary task",
                )
            ],
        ),
    )

    mock_client = AsyncMock()
    enhancer = PromptEnhancer(client=mock_client, analyzer=mock_analyzer)
    analysis, enhanced = await enhancer.enhance("help me")

    # Invariant: 1 analyzer call + 0 enhancer calls
    mock_analyzer.analyze.assert_called_once()
    mock_client.chat.assert_not_called()
    assert enhanced == ""
    assert analysis.ambiguity.needs_clarification is True


# --- Metamorphic Ambiguity Tests ---

@pytest.mark.parametrize(
    "phrasing1,phrasing2",
    [
        (
            "build me a web calculator using Vue",
            "using Vue, create a calculator webpage",
        ),
        (
            "write a professional email to my manager about vacation next week",
            "compose an email to my manager regarding next week vacation with professional tone",
        ),
    ],
)
@pytest.mark.asyncio
async def test_phase5_metamorphic_ambiguity_consistency(phrasing1, phrasing2):
    """Metamorphic: Equivalent phrasings produce consistent non-clarifying ambiguity assessments."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build calculator",
        "explicit_requirements": ["Vue", "calculator"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["style preferences"],
        "ambiguity": {
            "score": 15,
            "level": "LOW",
            "needs_clarification": False,
            "reasons": [],
            "questions": [],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res1 = await analyzer.analyze(phrasing1)
    res2 = await analyzer.analyze(phrasing2)

    assert res1.ambiguity.needs_clarification == res2.ambiguity.needs_clarification
    assert res1.ambiguity.level == res2.ambiguity.level
    assert res1.ambiguity.needs_clarification is False


# --- Cross-Domain Question Neutrality Tests (No Candidate Tech Injection) ---

@pytest.mark.asyncio
async def test_phase5_question_neutrality_no_injected_candidate_technologies():
    """Questions must never inject candidate technologies (e.g. React, PostgreSQL)."""
    mock_client = AsyncMock()
    # Model improperly emits candidate answers in question
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.5},
        "objective": "Build an app",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {
            "score": 85,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Unclear database"],
            "questions": [
                {
                    "id": "q1",
                    "question": "Which database do you want: PostgreSQL, MySQL, MongoDB or SQLite?",
                    "reason": "Database choice",
                    "target_dimension": "database",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res = await analyzer.analyze("make an app")

    assert res.ambiguity.needs_clarification is True
    assert len(res.ambiguity.questions) >= 1
    q_text = res.ambiguity.questions[0].question.lower()
    # Candidate technologies must have been neutralized
    assert "postgresql" not in q_text
    assert "mongodb" not in q_text
    assert "sqlite" not in q_text


# --- Phase 5 Architectural Correction Tests (Behavior Classes A through H) ---

@pytest.mark.asyncio
async def test_phase5_correction_class_a_complete_answer_proceeds_to_enhancement():
    """Class A: Complete answer resolves target dimension -> ambiguity drops, enhancement proceeds."""
    original_prompt = "build me a tool"
    answers = [
        ClarificationAnswer(
            question_id="q1",
            question="What is the primary objective and target platform of this tool?",
            answer="A command-line utility in Python to convert markdown tables into CSV format",
            target_dimension="primary objective and target platform",
        )
    ]

    mock_client = AsyncMock()
    # Re-analysis response from model incorporates answer and lowers ambiguity
    mock_client.chat.side_effect = [
        # Analyzer call
        json.dumps({
            "classification": {"category": "CODING", "confidence": 0.95},
            "objective": "Convert markdown tables to CSV format using a Python CLI tool",
            "explicit_requirements": ["Python", "command-line utility", "markdown tables to CSV"],
            "constraints": [],
            "preferences": [],
            "missing_context": [],
            "ambiguity": {
                "score": 15,
                "level": "LOW",
                "needs_clarification": False,
                "reasons": [],
                "questions": [],
            },
        }),
        # Enhancer call
        "OBJECTIVE:\nBuild a Python CLI tool that parses markdown tables and exports them to CSV format.",
    ]

    analyzer = PromptAnalyzer(client=mock_client)
    enhancer = PromptEnhancer(client=mock_client, analyzer=analyzer)

    analysis, enhanced_text = await enhancer.enhance(
        user_prompt=original_prompt,
        clarification_answers=answers,
    )

    assert analysis.ambiguity.needs_clarification is False
    assert analysis.ambiguity.level == AmbiguityLevel.LOW
    assert analysis.ambiguity.score < 30
    assert enhanced_text != ""
    assert "CLI tool" in enhanced_text or "markdown" in enhanced_text


@pytest.mark.asyncio
async def test_phase5_correction_class_b_partial_answer_leaves_critical_ambiguity_unresolved():
    """Class B: Partial answer resolves dimension A, but critical dimension B remains unresolved -> clarification remains active, score NOT forced LOW."""
    original_prompt = "build me a platform"
    # User answered only dimension A (primary objective), leaving critical dimension B (deployment environment)
    answers = [
        ClarificationAnswer(
            question_id="q1",
            question="What is the primary function of this platform?",
            answer="A real-time telemetry processing pipeline",
            target_dimension="primary function",
        )
    ]

    mock_client = AsyncMock()
    # Re-analysis: model still finds critical architecture ambiguity for deployment environment
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.85},
        "objective": "Build a real-time telemetry processing pipeline platform",
        "explicit_requirements": ["real-time telemetry processing pipeline"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["deployment environment"],
        "ambiguity": {
            "score": 70,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Target deployment environment fundamentally alters pipeline architecture."],
            "questions": [
                {
                    "id": "q1",
                    "question": "What is the target execution or deployment environment for this pipeline?",
                    "reason": "Architecture fundamentally differs between cloud, edge, and on-premise.",
                    "target_dimension": "deployment environment",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    enhancer = PromptEnhancer(client=mock_client, analyzer=analyzer)

    analysis, enhanced_text = await enhancer.enhance(
        user_prompt=original_prompt,
        clarification_answers=answers,
    )

    # Invariant: Score must NOT be forced LOW, clarification must NOT be falsely cleared
    assert analysis.ambiguity.needs_clarification is True
    assert analysis.ambiguity.score == 70
    assert analysis.ambiguity.level == AmbiguityLevel.HIGH
    assert len(analysis.ambiguity.questions) == 1
    assert analysis.ambiguity.questions[0].target_dimension == "deployment environment"
    # Enhancer LLM was not called
    assert enhanced_text == ""


@pytest.mark.asyncio
async def test_phase5_correction_class_c_vague_answer_does_not_force_low_ambiguity():
    """Class C: Vague answer ('whatever', 'I don't know') must NOT automatically force ambiguity LOW or clear questions."""
    original_prompt = "build me an app"
    answers = [
        ClarificationAnswer(
            question_id="q1",
            question="What is the primary purpose and platform of this app?",
            answer="whatever",
            target_dimension="primary purpose and platform",
        )
    ]

    mock_client = AsyncMock()
    # Re-analysis: model notes vague response leaves ambiguity unresolved
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.5},
        "objective": "Build an app",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["platform and domain"],
        "ambiguity": {
            "score": 80,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Vague clarification leaves primary purpose and platform unresolved."],
            "questions": [
                {
                    "id": "q1",
                    "question": "What problem or task should this application address?",
                    "reason": "Application architecture requires basic domain definition.",
                    "target_dimension": "primary domain and problem",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(original_prompt, clarification_answers=answers)

    # Invariant: Score is NOT clamped to <= 20, level is NOT forced to LOW
    assert analysis.ambiguity.needs_clarification is True
    assert analysis.ambiguity.score >= 65
    assert analysis.ambiguity.level == AmbiguityLevel.HIGH
    assert len(analysis.ambiguity.questions) >= 1


@pytest.mark.asyncio
async def test_phase5_correction_class_d_different_follow_up_question_without_repeating_resolved():
    """Class D: When dimension A is resolved, system may ask about separate critical dimension B without repeating A."""
    original_prompt = "create a plan"
    answers = [
        ClarificationAnswer(
            question_id="q1",
            question="What is the objective of the plan?",
            answer="Prepare for an upcoming marathon race",
            target_dimension="plan objective",
        )
    ]

    mock_client = AsyncMock()
    # Re-analysis asks about timeframe (dimension B) without re-asking plan objective (dimension A)
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "PLANNING", "confidence": 0.9},
        "objective": "Create a marathon training preparation plan",
        "explicit_requirements": ["marathon training plan"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["timeline duration"],
        "ambiguity": {
            "score": 68,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Training schedule requires duration/timeframe to structure milestones."],
            "questions": [
                {
                    "id": "q1",
                    "question": "How many weeks or months do you have to train before the race?",
                    "reason": "Duration dictates progression schedule.",
                    "target_dimension": "timeline duration",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(original_prompt, clarification_answers=answers)

    assert analysis.ambiguity.needs_clarification is True
    assert len(analysis.ambiguity.questions) == 1
    # Dimension A is NOT repeated; Dimension B is asked
    assert "objective" not in analysis.ambiguity.questions[0].target_dimension.lower()
    assert "duration" in analysis.ambiguity.questions[0].target_dimension.lower()


@pytest.mark.asyncio
async def test_phase5_correction_class_e_duplicate_question_suppressed():
    """Class E: Analyzer stochastically re-asking an already answered dimension has the duplicate suppressed."""
    original_prompt = "build me a service"
    answers = [
        ClarificationAnswer(
            question_id="q1",
            question="What protocol or interface should this service expose?",
            answer="A gRPC API with protobuf contracts",
            target_dimension="protocol and interface",
        )
    ]

    mock_client = AsyncMock()
    # Analyzer attempts to ask the same interface dimension again
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.9},
        "objective": "Build a service with gRPC API",
        "explicit_requirements": ["gRPC API", "protobuf contracts"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {
            "score": 65,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Interface choice."],
            "questions": [
                {
                    "id": "q1",
                    "question": "What protocol or interface should this service use?",
                    "reason": "Protocol choice.",
                    "target_dimension": "protocol and interface",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(original_prompt, clarification_answers=answers)

    # Invariant: Duplicate question targeting already-answered dimension is suppressed
    assert len(analysis.ambiguity.questions) == 0
    assert analysis.ambiguity.needs_clarification is False
    # Score is NOT artificially clamped to <= 20
    assert analysis.ambiguity.score == 65


@pytest.mark.asyncio
async def test_phase5_correction_class_f_reasonable_assumptions_preserves_truthful_high_score():
    """Class F: allow_reasonable_assumptions=True proceeds while score and level remain truthful (not falsified)."""
    original_prompt = "write a document"

    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.7),
        objective="Write a document",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["document topic and audience"],
        ambiguity=AmbiguityAssessment(
            score=75,
            level=AmbiguityLevel.HIGH,
            needs_clarification=False,  # Bypassed by user authorization
            reasons=["Unspecified document topic and target audience."],
            questions=[],
        ),
    )

    mock_client = AsyncMock()
    mock_client.chat.return_value = "OBJECTIVE:\nDraft a general overview document based on standard conventions."

    enhancer = PromptEnhancer(client=mock_client, analyzer=mock_analyzer)
    analysis, enhanced_text = await enhancer.enhance(
        user_prompt=original_prompt,
        allow_reasonable_assumptions=True,
    )

    # Invariant: Workflow proceeded to enhancement
    assert enhanced_text != ""
    mock_client.chat.assert_called_once()
    # Invariant: Score remains truthful (75, HIGH), not falsified to 20 or LOW
    assert analysis.ambiguity.score == 75
    assert analysis.ambiguity.level == AmbiguityLevel.HIGH
    assert analysis.ambiguity.needs_clarification is False


@pytest.mark.asyncio
async def test_phase5_correction_class_g_empty_or_whitespace_answer_rejected():
    """Class G: Empty or whitespace-only answers are rejected and do not count as resolved context."""
    original_prompt = "analyze data"
    answers = [
        ClarificationAnswer(
            question_id="q1",
            question="What dataset should be analyzed?",
            answer="     ",
            target_dimension="dataset source",
        )
    ]

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "DATA_ANALYSIS", "confidence": 0.5},
        "objective": "Analyze data",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["dataset source"],
        "ambiguity": {
            "score": 85,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Unspecified dataset."],
            "questions": [
                {
                    "id": "q1",
                    "question": "What dataset or metrics should be analyzed?",
                    "reason": "Analysis requires a data source.",
                    "target_dimension": "dataset source",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    analysis = await analyzer.analyze(original_prompt, clarification_answers=answers)

    # Invariant: Empty answer did NOT resolve ambiguity or clear questions
    assert analysis.ambiguity.needs_clarification is True
    assert analysis.ambiguity.score >= 65
    assert len(analysis.ambiguity.questions) >= 1


@pytest.mark.asyncio
async def test_phase5_correction_class_h_call_budget_invariants():
    """Class H: Strict call budget verification across clarification and enhancement states."""
    # 1. Initial clarification request: 1 analyzer call + 0 enhancer calls
    mock_analyzer_1 = AsyncMock()
    mock_analyzer_1.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.5),
        objective="Build a system",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["system domain"],
        ambiguity=AmbiguityAssessment(
            score=85,
            level=AmbiguityLevel.HIGH,
            needs_clarification=True,
            reasons=["Unspecified domain"],
            questions=[ClarificationQuestion(id="q1", question="What domain?", reason="r", target_dimension="domain")],
        ),
    )
    mock_client_1 = AsyncMock()
    enhancer_1 = PromptEnhancer(client=mock_client_1, analyzer=mock_analyzer_1)
    await enhancer_1.enhance("build a system")
    mock_analyzer_1.analyze.assert_called_once()
    mock_client_1.chat.assert_not_called()

    # 2. Resolved follow-up: 1 analyzer call + 1 enhancer call
    mock_analyzer_2 = AsyncMock()
    mock_analyzer_2.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build an inventory system",
        explicit_requirements=["inventory system"],
        constraints=[],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(
            score=20,
            level=AmbiguityLevel.LOW,
            needs_clarification=False,
            reasons=[],
            questions=[],
        ),
    )
    mock_client_2 = AsyncMock()
    mock_client_2.chat.return_value = "OBJECTIVE:\nBuild an inventory system."
    enhancer_2 = PromptEnhancer(client=mock_client_2, analyzer=mock_analyzer_2)
    await enhancer_2.enhance(
        "build a system",
        clarification_answers=[ClarificationAnswer(question_id="q1", question="What domain?", answer="Inventory", target_dimension="domain")],
    )
    mock_analyzer_2.analyze.assert_called_once()
    mock_client_2.chat.assert_called_once()

    # 3. Further clarification follow-up (unresolved partial/vague): 1 analyzer call + 0 enhancer calls
    mock_analyzer_3 = AsyncMock()
    mock_analyzer_3.analyze.return_value = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.6),
        objective="Build a system",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["database storage"],
        ambiguity=AmbiguityAssessment(
            score=70,
            level=AmbiguityLevel.HIGH,
            needs_clarification=True,
            reasons=["Storage choice"],
            questions=[ClarificationQuestion(id="q1", question="What database?", reason="r", target_dimension="database")],
        ),
    )
    mock_client_3 = AsyncMock()
    enhancer_3 = PromptEnhancer(client=mock_client_3, analyzer=mock_analyzer_3)
    await enhancer_3.enhance(
        "build a system",
        clarification_answers=[ClarificationAnswer(question_id="q1", question="What domain?", answer="whatever", target_dimension="domain")],
    )
    mock_analyzer_3.analyze.assert_called_once()
    mock_client_3.chat.assert_not_called()


# ==============================================================================
# PHASE 6 — GENERAL INTENT-PRESERVATION VALIDATOR TESTS
# ==============================================================================

# --- Group A: Perfect Preservation ---

@pytest.mark.asyncio
@pytest.mark.parametrize("category,objective,reqs,prompt_text", [
    (PromptCategory.CODING, "Build a CLI CSV parser", ["parse CSV", "output JSON"], "Create a Python CLI script to parse CSV files and export JSON."),
    (PromptCategory.WRITING, "Draft a project status email", ["status update", "next sprint goals"], "Write a concise project status email to the team."),
    (PromptCategory.EDUCATION, "Explain binary search", ["step-by-step algorithm", "time complexity"], "Explain the binary search algorithm simply with complexity analysis."),
    (PromptCategory.RESEARCH, "Compare SQL vs NoSQL databases", ["pros and cons", "tabular summary"], "Conduct a comparative analysis of SQL vs NoSQL with a comparison table."),
])
async def test_phase6_group_a_perfect_preservation(category, objective, reqs, prompt_text):
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=category, confidence=0.9),
        objective=objective,
        explicit_requirements=reqs,
        constraints=[],
        preferences=["clear and concise"],
        missing_context=[],
    )

    validation = await validator.validate(
        original_prompt=prompt_text,
        analysis=analysis,
        enhanced_prompt=f"OBJECTIVE: {objective}\nREQUIREMENTS: " + ", ".join(reqs),
    )

    assert validation.is_valid is True
    assert len([i for i in validation.issues if i.severity == ValidationSeverity.ERROR]) == 0


# --- Group B: Missing Explicit Requirement ---

@pytest.mark.asyncio
async def test_phase6_group_b_missing_explicit_requirement():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": False,
        "issues": [{
            "type": "MISSING_REQUIREMENT",
            "severity": "ERROR",
            "message": "The explicit deliverable 'export JSON' was completely omitted.",
            "evidence": "export JSON",
        }],
    })
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Parse CSV and export JSON",
        explicit_requirements=["parse CSV", "export JSON"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    # Enhanced prompt only mentions CSV, dropping JSON export
    enhanced = "OBJECTIVE: Parse CSV files and print results to terminal."
    validation = await validator.validate(
        original_prompt="parse CSV and export JSON",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    error_issues = [i for i in validation.issues if i.severity == ValidationSeverity.ERROR]
    assert any(i.type == ValidationIssueType.MISSING_REQUIREMENT for i in error_issues)


# --- Group C: Compound Task Loss ---

@pytest.mark.asyncio
async def test_phase6_group_c_compound_task_loss():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": False,
        "issues": [{
            "type": "COMPOUND_TASK_LOSS",
            "severity": "ERROR",
            "message": "User requested compound deliverables (summary, risk matrix, roadmap), but only summary was included.",
            "evidence": "risk matrix, mitigation roadmap",
        }],
    })
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.PLANNING, confidence=0.9),
        objective="Create project plan with summary, risk matrix, and mitigation roadmap",
        explicit_requirements=["executive summary", "risk matrix", "mitigation roadmap"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    enhanced = "Write an executive summary of the migration plan."
    validation = await validator.validate(
        original_prompt="give me an executive summary, a risk matrix, and a mitigation roadmap",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    assert any(i.type == ValidationIssueType.COMPOUND_TASK_LOSS for i in validation.issues)


# --- Group D: Quantity Mismatch ---

@pytest.mark.asyncio
async def test_phase6_group_d_quantity_mismatch_deterministic_and_llm():
    # Deterministic check catches mismatched count (5 requested vs 3 provided)
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CREATIVE, confidence=0.9),
        objective="Provide 5 blog title ideas",
        explicit_requirements=["5 blog title ideas"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    # User wanted 5, prompt explicitly says 3 ideas
    enhanced = "OBJECTIVE: Brainstorm ideas.\nProvide 3 blog title ideas for the new product launch."
    validation = await validator.validate(
        original_prompt="give me 5 blog title ideas",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    assert any(i.type == ValidationIssueType.QUANTITY_MISMATCH for i in validation.issues)


# --- Group E: Invented Requirement ---

@pytest.mark.asyncio
async def test_phase6_group_e_invented_requirement():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": False,
        "issues": [{
            "type": "INVENTED_REQUIREMENT",
            "severity": "ERROR",
            "message": "Invented biometric login and cloud data synchronization not requested by user.",
            "evidence": "biometric login, encrypted cloud sync",
        }],
    })
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build a local notes app",
        explicit_requirements=["notes app"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    enhanced = "Build a notes app. CORE REQUIREMENTS: Must implement biometric login and encrypted cloud sync."
    validation = await validator.validate(
        original_prompt="build a local notes app",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    assert any(i.type == ValidationIssueType.INVENTED_REQUIREMENT for i in validation.issues)


# --- Group F: Invented Technology ---

@pytest.mark.asyncio
async def test_phase6_group_f_invented_technology():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": False,
        "issues": [{
            "type": "INVENTED_TECHNOLOGY",
            "severity": "ERROR",
            "message": "Prompt mandates Apache Spark and PostgreSQL when user requested a generic sorting script.",
            "evidence": "Apache Spark, PostgreSQL",
        }],
    })
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Sort files by date",
        explicit_requirements=["sort files by date"],
        constraints=[],
        preferences=[],
        missing_context=["programming language"],
    )

    enhanced = "Write a file sorting utility. You must implement this using Apache Spark and PostgreSQL."
    validation = await validator.validate(
        original_prompt="make a script to sort files by date",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    assert any(i.type == ValidationIssueType.INVENTED_TECHNOLOGY for i in validation.issues)


# --- Group G: Negative Constraint Violation ---

@pytest.mark.asyncio
async def test_phase6_group_g_negative_constraint_violation():
    # Deterministic check catches direct contradiction of "without recursion"
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Traverse binary tree without recursion",
        explicit_requirements=["binary tree traversal"],
        constraints=["do not use recursion"],
        preferences=[],
        missing_context=[],
    )

    enhanced = "Write a binary tree traversal algorithm. You must implement with recursion."
    validation = await validator.validate(
        original_prompt="traverse binary tree without recursion",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    assert any(i.type == ValidationIssueType.CONSTRAINT_CONTRADICTION for i in validation.issues)


# --- Group H: Rejected Choice Reintroduced ---

@pytest.mark.asyncio
async def test_phase6_group_h_rejected_choice_reintroduced():
    # Deterministic check catches reintroduced superseded choice (calculator rejected for quiz)
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Create a shoe recommendation quiz",
        explicit_requirements=["quiz"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    enhanced = "PRIMARY OBJECTIVE: Build a calculator for the website to recommend products."
    validation = await validator.validate(
        original_prompt="maybe a calculator? no like a quiz for shoes",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    assert any(i.type == ValidationIssueType.REJECTED_OPTION_REINTRODUCED for i in validation.issues)


# --- Group I: Missing Context Promotion ---

@pytest.mark.asyncio
async def test_phase6_group_i_missing_context_promotion():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": False,
        "issues": [{
            "type": "MISSING_CONTEXT_PROMOTED",
            "severity": "ERROR",
            "message": "Missing context 'data storage method' was promoted to mandatory MongoDB requirement.",
            "evidence": "MongoDB",
        }],
    })
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build attendance app",
        explicit_requirements=["attendance tracking"],
        constraints=[],
        preferences=[],
        missing_context=["data storage method"],
    )

    enhanced = "Build an attendance tracking app. CORE REQUIREMENTS: All data must be saved in MongoDB."
    validation = await validator.validate(
        original_prompt="make attendance app",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    assert validation.is_valid is False
    assert any(i.type == ValidationIssueType.MISSING_CONTEXT_PROMOTED for i in validation.issues)


# --- Group J: Preference Loss Policy ---

@pytest.mark.asyncio
async def test_phase6_group_j_preference_loss_policy():
    # Documented policy: Non-critical style preference dilution flags a WARNING, which does NOT invalidate output
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": True,
        "issues": [{
            "type": "MISSING_PREFERENCE",
            "severity": "WARNING",
            "message": "Tone preference for concise style could be emphasized more clearly.",
            "evidence": "concise style",
        }],
    })
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.9),
        objective="Draft vacation notice",
        explicit_requirements=["vacation notice"],
        constraints=[],
        preferences=["concise style"],
        missing_context=[],
    )

    enhanced = "Draft an out of office vacation notice explaining dates."
    validation = await validator.validate(
        original_prompt="draft a concise vacation notice",
        analysis=analysis,
        enhanced_prompt=enhanced,
    )

    # WARNING alone does NOT fail validation
    assert validation.is_valid is True
    assert any(i.type == ValidationIssueType.MISSING_PREFERENCE and i.severity == ValidationSeverity.WARNING for i in validation.issues)


# --- Group K: Reasonable Assumptions Handling ---

@pytest.mark.asyncio
async def test_phase6_group_k_reasonable_assumptions():
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "is_valid": True,
        "issues": [{
            "type": "OTHER",
            "severity": "INFO",
            "message": "Minimal stated assumption appropriately labeled.",
            "evidence": "choose a simple file-based format and state assumption",
        }],
    })
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build CLI expense tracker",
        explicit_requirements=["CLI expense tracker"],
        constraints=[],
        preferences=[],
        missing_context=["storage method"],
    )

    enhanced = (
        "Build a CLI expense tracker. If storage mechanism is unspecified, choose a simple "
        "file-based format (such as JSON) and state the assumption before proceeding."
    )
    validation = await validator.validate(
        original_prompt="build cli expense tracker",
        analysis=analysis,
        enhanced_prompt=enhanced,
        allow_reasonable_assumptions=True,
    )

    assert validation.is_valid is True
    assert len([i for i in validation.issues if i.severity == ValidationSeverity.ERROR]) == 0


# --- Group L: Automatic Repair ---

@pytest.mark.asyncio
async def test_phase6_group_l_automatic_repair_success():
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CREATIVE, confidence=0.9),
        objective="Provide 5 blog title ideas",
        explicit_requirements=["5 blog title ideas"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = analysis

    # Initial enhancer produces 3 ideas (invalid: quantity mismatch)
    initial_invalid_enhancement = "Provide 3 blog title ideas for the product."
    # Repair produces 5 ideas (valid)
    repaired_valid_enhancement = "Provide 5 creative blog title ideas for the product."

    mock_enhancer_client = AsyncMock()
    mock_enhancer_client.chat.side_effect = [
        initial_invalid_enhancement,  # call 1: initial enhance
        repaired_valid_enhancement,   # call 2: repair enhance
    ]

    mock_validator_client = AsyncMock()
    mock_validator_client.chat.side_effect = [
        json.dumps({
            "is_valid": False,
            "issues": [{
                "type": "QUANTITY_MISMATCH",
                "severity": "ERROR",
                "message": "Requested 5 ideas, got 3.",
                "evidence": "3 blog title ideas",
            }],
        }),  # validation 1
        json.dumps({"is_valid": True, "issues": []}),  # validation 2 (after repair)
    ]
    validator = PromptValidator(client=mock_validator_client)

    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=mock_analyzer,
        validator=validator,
    )

    result = await enhancer.enhance("give me 5 blog title ideas", validate=True)

    assert result.repair_attempted is True
    assert result.validation.is_valid is True
    assert "5" in result.enhanced_prompt
    assert mock_enhancer_client.chat.call_count == 2
    assert mock_validator_client.chat.call_count == 2


# --- Group M: Repair Failure ---

@pytest.mark.asyncio
async def test_phase6_group_m_repair_failure_clean_response():
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Sort images by date",
        explicit_requirements=["sort images by date"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = analysis

    # Initial and repaired both invent Apache Spark
    initial_enhancement = "Sort images. Must use Apache Spark."
    repaired_enhancement = "Sort images. Still requires Apache Spark."

    mock_enhancer_client = AsyncMock()
    mock_enhancer_client.chat.side_effect = [
        initial_enhancement,
        repaired_enhancement,
    ]

    mock_validator_client = AsyncMock()
    mock_validator_client.chat.side_effect = [
        json.dumps({
            "is_valid": False,
            "issues": [{
                "type": "INVENTED_TECHNOLOGY",
                "severity": "ERROR",
                "message": "Invented Apache Spark.",
                "evidence": "Apache Spark",
            }],
        }),
        json.dumps({
            "is_valid": False,
            "issues": [{
                "type": "INVENTED_TECHNOLOGY",
                "severity": "ERROR",
                "message": "Still invented Apache Spark after repair.",
                "evidence": "Apache Spark",
            }],
        }),
    ]
    validator = PromptValidator(client=mock_validator_client)

    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=mock_analyzer,
        validator=validator,
    )

    result = await enhancer.enhance("make script to sort images", validate=True)

    # Must NOT enter 3rd repair loop; result is invalid
    assert result.repair_attempted is True
    assert result.validation.is_valid is False
    assert mock_enhancer_client.chat.call_count == 2
    assert mock_validator_client.chat.call_count == 2


def test_phase6_group_m_api_returns_validation_failed_status():
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Sort images",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    failed_val = PromptValidation(
        is_valid=False,
        issues=[ValidationIssue(
            type=ValidationIssueType.INVENTED_TECHNOLOGY,
            severity=ValidationSeverity.ERROR,
            message="Unrecoverable invented technology.",
            evidence="Apache Spark",
        )],
    )

    from app.core.enhancer import EnhanceResult
    fake_result = EnhanceResult(
        analysis=analysis,
        enhanced_prompt="Corrupted prompt",
        validation=failed_val,
        repair_attempted=True,
    )

    with patch("app.api.routes.prompt_enhancer.enhance", new_callable=AsyncMock) as mock_enhance:
        mock_enhance.return_value = fake_result
        response = client.post("/api/enhance", json={"prompt": "sort images"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "validation_failed"
        assert "validation" in data
        assert data["validation"]["is_valid"] is False
        assert "could not produce a sufficiently faithful enhanced prompt" in data["message"]


# --- Group N: Call Budget Invariants ---

@pytest.mark.asyncio
async def test_phase6_group_n_call_budget_invariants():
    # Invariant 1: First-pass valid clear request = 1 analyzer + 1 enhancer + 1 validator = 3 calls
    analysis_clear = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build a CLI tool",
        explicit_requirements=["CLI tool"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = analysis_clear

    mock_enhancer_client = AsyncMock()
    mock_enhancer_client.chat.return_value = "Build a Python CLI tool."

    mock_validator_client = AsyncMock()
    mock_validator_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_validator_client)

    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=mock_analyzer,
        validator=validator,
    )

    await enhancer.enhance("build a CLI tool", validate=True)
    assert mock_analyzer.analyze.call_count == 1
    assert mock_enhancer_client.chat.call_count == 1
    assert mock_validator_client.chat.call_count == 1
    # Total = 1 + 1 + 1 = 3 LLM calls

    # Invariant 2: Repaired request = 1 analyzer + 2 enhancer + 2 validator = 5 calls max
    mock_analyzer.analyze.reset_mock()
    mock_enhancer_client.chat.reset_mock()
    mock_validator_client.chat.reset_mock()

    mock_enhancer_client.chat.side_effect = ["Initial draft", "Repaired draft"]
    mock_validator_client.chat.side_effect = [
        json.dumps({"is_valid": False, "issues": [{"type": "MISSING_REQUIREMENT", "severity": "ERROR", "message": "Missing req", "evidence": None}]}),
        json.dumps({"is_valid": True, "issues": []}),
    ]

    await enhancer.enhance("build a CLI tool", validate=True)
    assert mock_analyzer.analyze.call_count == 1
    assert mock_enhancer_client.chat.call_count == 2
    assert mock_validator_client.chat.call_count == 2
    # Total = 1 + 2 + 2 = 5 LLM calls


# --- Metamorphic Validation Tests ---

@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_a,prompt_b", [
    ("Draft 4 announcement posts for our upcoming webinar on social media", "Compose 4 social media posts announcing our webinar"),
    ("Write a Python script to convert markdown to PDF without external APIs", "Without using external APIs, create a script in Python that converts markdown files into PDF"),
])
async def test_phase6_metamorphic_validation_consistency(prompt_a, prompt_b):
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_client)

    if "markdown" in prompt_a.lower():
        analysis_a = PromptAnalysis(
            classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
            objective="Write a Python script to convert markdown to PDF without external APIs",
            explicit_requirements=["Python", "convert markdown to PDF"],
            constraints=["do not use external APIs"],
            preferences=[],
            missing_context=[],
        )
        analysis_b = PromptAnalysis(
            classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
            objective="Create a Python script that converts markdown files into PDF without external APIs",
            explicit_requirements=["Python", "convert markdown to PDF"],
            constraints=["do not use external APIs"],
            preferences=[],
            missing_context=[],
        )
        enh_a = "Create a Python script to convert Markdown files to PDF without using external APIs."
        enh_b = "Create a Python script that converts markdown files into PDF without using external APIs."
    else:
        analysis_a = PromptAnalysis(
            classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.9),
            objective="Draft 4 webinar announcement posts",
            explicit_requirements=["4 posts", "webinar announcement"],
            constraints=[],
            preferences=[],
            missing_context=[],
        )
        analysis_b = PromptAnalysis(
            classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.9),
            objective="Compose 4 social media announcement posts",
            explicit_requirements=["4 posts", "webinar announcement"],
            constraints=[],
            preferences=[],
            missing_context=[],
        )
        enh_a = "Write 4 social media posts announcing the upcoming webinar."
        enh_b = "Compose 4 social media posts announcing the upcoming webinar."

    val_a = await validator.validate(
        original_prompt=prompt_a,
        analysis=analysis_a,
        enhanced_prompt=enh_a,
    )
    val_b = await validator.validate(
        original_prompt=prompt_b,
        analysis=analysis_b,
        enhanced_prompt=enh_b,
    )

    assert val_a.is_valid == val_b.is_valid == True


# ==============================================================================
# PHASE 7 — EXPLAINABLE PROMPT QUALITY SCORING TESTS
# ==============================================================================

# --- Helper to mock semantic scorer output ---
def make_mock_semantic_json(clarity=90, specificity=90, output_def=90, structure=90, concision=90):
    return json.dumps({
        "clarity": {"score": clarity, "reason": "Clear instructions."},
        "specificity": {"score": specificity, "reason": "Appropriate concrete details."},
        "output_definition": {"score": output_def, "reason": "Deliverable clearly defined."},
        "structure": {"score": structure, "reason": "Logical organization suited to complexity."},
        "concision": {"score": concision, "reason": "Controlled length without fluff."},
    })


# --- Group A: High Quality ---

@pytest.mark.asyncio
@pytest.mark.parametrize("category,orig,enh", [
    (PromptCategory.CODING, "build python markdown parser", "Act as an experienced software developer. Create a Python CLI tool to parse Markdown files and output structured AST JSON.\n\nCORE REQUIREMENTS:\n- Python\n- Parse Markdown\n- Export AST JSON\n\nOUTPUT EXPECTATIONS:\nProvide clean, modular code with usage examples."),
    (PromptCategory.WRITING, "draft product launch email", "Act as a professional communications writer. Draft an engaging product launch email announcing our new mobile app release.\n\nCORE REQUIREMENTS:\n- Clear subject line\n- Highlight key features\n- Call to action to download\n\nOUTPUT EXPECTATIONS:\nStructured email format ready for review."),
])
async def test_phase7_group_a_high_quality(category, orig, enh):
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(95, 90, 95, 95, 90)
    scorer = PromptScorer(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=category, confidence=0.9),
        objective=orig,
        explicit_requirements=[r.strip() for r in orig.split()[1:]],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    val = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score(orig, analysis, enh, val)
    assert report.overall_score >= 85
    assert report.overall_label in {"Strong", "Excellent"}
    assert report.intent_preservation.score == 100
    assert report.requirement_coverage.score >= 85


# --- Group B: Clear but Under-Specified ---

@pytest.mark.asyncio
async def test_phase7_group_b_clear_but_underspecified():
    # Model returns low clarity, specificity and output definition scores
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(clarity=65, specificity=55, output_def=50, structure=70, concision=80)
    scorer = PromptScorer(client=mock_client)

    orig = "build an inventory management tool"
    enh = "Build an inventory tool. Keep track of items."
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.85),
        objective="Build inventory tool",
        explicit_requirements=["inventory tool"],
        constraints=[],
        preferences=[],
        missing_context=["storage method", "interface"],
    )
    val = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score(orig, analysis, enh, val)
    assert report.specificity.score <= 60
    assert report.output_definition.score <= 55
    # Overall score reflects lower dimensional scores
    assert report.overall_score <= 85


# --- Group C: Over-Inflated Prompt ---

@pytest.mark.asyncio
async def test_phase7_group_c_over_inflated_prompt():
    # Long repetitive enhanced text for a short request
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(clarity=80, specificity=80, output_def=80, structure=75, concision=55)
    scorer = PromptScorer(client=mock_client)

    orig = "write a function to add two numbers"
    # Artificially bloated prompt with repeated filler
    enh = "Act as an experienced developer. " + "Write a function that adds two numbers. Ensure that addition is performed correctly. " * 30
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Write add function",
        explicit_requirements=["add two numbers"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    val = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score(orig, analysis, enh, val)
    assert report.concision.score <= 65
    assert report.inflation_risk in {InflationRisk.MEDIUM, InflationRisk.HIGH}
    assert report.inflation_ratio > 5.0


# --- Group D: Already Good Input ---

@pytest.mark.asyncio
async def test_phase7_group_d_already_good_input():
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(clarity=95, specificity=95, output_def=95, structure=95, concision=95)
    scorer = PromptScorer(client=mock_client)

    orig = (
        "Write a Python FastAPI service that exposes a POST /items endpoint accepting an item name and price, "
        "validating with Pydantic, and persisting into an SQLite database with SQLAlchemy."
    )
    enh = (
        "Act as an experienced Python developer.\n"
        "Implement a FastAPI microservice providing a POST /items endpoint.\n\n"
        "CORE REQUIREMENTS:\n"
        "- Validate incoming request payloads (name and price) using Pydantic.\n"
        "- Persist validated records to an SQLite database using SQLAlchemy.\n\n"
        "OUTPUT EXPECTATIONS:\n"
        "Provide modular FastAPI router code, Pydantic schemas, and SQLAlchemy database models."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create FastAPI items endpoint with SQLite",
        explicit_requirements=["FastAPI", "Pydantic", "SQLite", "SQLAlchemy"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    val = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score(orig, analysis, enh, val)
    assert report.overall_score >= 90
    assert report.inflation_risk == InflationRisk.LOW
    assert report.inflation_ratio < 3.0


# --- Group E: Simple Task (No Heading Bias) ---

@pytest.mark.asyncio
async def test_phase7_group_e_simple_task_no_heading_bias():
    # Prompt is concise with NO section headings; should receive high structure score
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(clarity=95, specificity=90, output_def=90, structure=95, concision=95)
    scorer = PromptScorer(client=mock_client)

    orig = "what is the capital of France"
    enh = "State the official capital city of France concisely in one sentence."
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.GENERAL, confidence=0.9),
        objective="Identify capital of France",
        explicit_requirements=[],
        constraints=[],
        preferences=["concise"],
        missing_context=[],
    )
    val = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score(orig, analysis, enh, val)
    assert report.structure.score >= 90
    assert report.concision.score >= 90
    assert report.overall_score >= 90


# --- Group F: Complex Task ---

@pytest.mark.asyncio
async def test_phase7_group_f_complex_task():
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(clarity=90, specificity=95, output_def=95, structure=95, concision=85)
    scorer = PromptScorer(client=mock_client)

    orig = "plan a 3-stage cloud migration for our legacy billing service, with risk matrix, rollback strategy, and team RACI"
    enh = (
        "Act as a cloud solutions architect. Design a comprehensive 3-stage cloud migration plan for a legacy billing service.\n\n"
        "CORE DELIVERABLES:\n"
        "1. 3-stage migration architecture and timeline.\n"
        "2. Detailed risk assessment and mitigation matrix.\n"
        "3. Automated and manual rollback strategies.\n"
        "4. Stakeholder RACI matrix for cross-functional teams.\n\n"
        "OUTPUT EXPECTATIONS:\n"
        "Provide a structured engineering document with clear operational gates."
    )
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.PLANNING, confidence=0.9),
        objective="Cloud migration plan with risk matrix, rollback, and RACI",
        explicit_requirements=["3-stage migration", "risk matrix", "rollback strategy", "RACI"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    val = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score(orig, analysis, enh, val)
    assert report.structure.score >= 90
    assert report.output_definition.score >= 90
    assert report.requirement_coverage.score == 100


# --- Group G: Requirement Coverage ---

@pytest.mark.asyncio
async def test_phase7_group_g_requirement_coverage_scoring():
    scorer = PromptScorer()

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Process files",
        explicit_requirements=["parse CSV", "filter active users", "export JSON"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    # All requirements present
    enh_full = "Write a tool to parse CSV files, filter active users, and export JSON."
    dim_full = scorer.compute_requirement_coverage(analysis, enh_full)
    assert dim_full.score == 100

    # Only 1 of 3 requirements present
    enh_partial = "Write a tool to parse CSV files."
    dim_partial = scorer.compute_requirement_coverage(analysis, enh_partial)
    assert dim_partial.score < 80


# --- Group H: Constraint Preservation ---

@pytest.mark.asyncio
async def test_phase7_group_h_constraint_preservation_scoring():
    scorer = PromptScorer()

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Traverse tree",
        explicit_requirements=["traverse tree"],
        constraints=["do not use recursion"],
        preferences=[],
        missing_context=[],
    )

    # Constraint preserved and validation has no contradiction
    val_clean = PromptValidation(is_valid=True, issues=[])
    dim_clean = scorer.compute_constraint_preservation(analysis, "Traverse tree without recursion.", val_clean)
    assert dim_clean.score == 100

    # Constraint contradicted in validator
    val_contradicted = PromptValidation(
        is_valid=False,
        issues=[ValidationIssue(
            type=ValidationIssueType.CONSTRAINT_CONTRADICTION,
            severity=ValidationSeverity.ERROR,
            message="Contradicted negative constraint.",
        )],
    )
    dim_violated = scorer.compute_constraint_preservation(analysis, "Traverse tree using recursion.", val_contradicted)
    assert dim_violated.score <= 50


# --- Group I: Inflation Diagnostics ---

def test_phase7_group_i_inflation_diagnostics():
    scorer = PromptScorer()

    # Short prompt expansion
    short_orig = "build app"
    normal_enh = "Build an application that manages daily task lists cleanly."  # ~9 words
    ratio_1, risk_1 = scorer.compute_inflation_diagnostic(short_orig, normal_enh)
    assert risk_1 == InflationRisk.LOW

    # Moderate prompt expansion
    med_orig = "create a personal portfolio website in HTML and CSS"
    med_enh = "Act as a web developer. Build a portfolio website in HTML and CSS."  # ~13 words
    ratio_2, risk_2 = scorer.compute_inflation_diagnostic(med_orig, med_enh)
    assert risk_2 == InflationRisk.LOW

    # Extreme prompt expansion
    huge_enh = "Build a portfolio. " * 100  # 300 words
    ratio_3, risk_3 = scorer.compute_inflation_diagnostic(med_orig, huge_enh)
    assert risk_3 == InflationRisk.HIGH


# --- Group J: Deterministic Overall Score Formula ---

@pytest.mark.asyncio
async def test_phase7_group_j_deterministic_overall_score_formula():
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(clarity=80, specificity=85, output_def=90, structure=75, concision=70)
    scorer = PromptScorer(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build tool",
        explicit_requirements=["build tool"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    val = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score("build tool", analysis, "Build tool properly.", val)

    # Compute expected score strictly from centralized weights
    expected = round(
        QUALITY_WEIGHTS["intent_preservation"] * report.intent_preservation.score
        + QUALITY_WEIGHTS["requirement_coverage"] * report.requirement_coverage.score
        + QUALITY_WEIGHTS["constraint_preservation"] * report.constraint_preservation.score
        + QUALITY_WEIGHTS["clarity"] * report.clarity.score
        + QUALITY_WEIGHTS["specificity"] * report.specificity.score
        + QUALITY_WEIGHTS["output_definition"] * report.output_definition.score
        + QUALITY_WEIGHTS["structure"] * report.structure.score
        + QUALITY_WEIGHTS["concision"] * report.concision.score
    )
    assert report.overall_score == expected
    assert report.overall_label == get_score_label(expected)


# --- Group K: Invalid Prompts Do Not Receive Quality Score ---

@pytest.mark.asyncio
async def test_phase7_group_k_invalid_prompt_no_quality_scoring():
    # If validation fails permanently, quality is None
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Sort images",
        explicit_requirements=["sort images"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = analysis

    mock_enhancer_client = AsyncMock()
    mock_enhancer_client.chat.side_effect = ["draft 1", "draft 2"]

    mock_validator_client = AsyncMock()
    mock_validator_client.chat.side_effect = [
        json.dumps({"is_valid": False, "issues": [{"type": "INVENTED_TECHNOLOGY", "severity": "ERROR", "message": "error", "evidence": None}]}),
        json.dumps({"is_valid": False, "issues": [{"type": "INVENTED_TECHNOLOGY", "severity": "ERROR", "message": "error", "evidence": None}]}),
    ]
    validator = PromptValidator(client=mock_validator_client)

    mock_scorer = AsyncMock()
    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=mock_analyzer,
        validator=validator,
        scorer=mock_scorer,
    )

    result = await enhancer.enhance("sort images", validate=True)
    assert result.validation.is_valid is False
    assert result.quality is None
    mock_scorer.score.assert_not_called()


# --- Group L: Repair Path Scores Repaired Version Only ---

@pytest.mark.asyncio
async def test_phase7_group_l_repair_path_scores_repaired_version_only():
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="5 ideas",
        explicit_requirements=["5 ideas"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = analysis

    mock_enhancer_client = AsyncMock()
    mock_enhancer_client.chat.side_effect = [
        "3 ideas",  # invalid initial
        "5 ideas",  # valid repair
    ]

    mock_validator_client = AsyncMock()
    mock_validator_client.chat.side_effect = [
        json.dumps({"is_valid": False, "issues": [{"type": "QUANTITY_MISMATCH", "severity": "ERROR", "message": "error", "evidence": None}]}),
        json.dumps({"is_valid": True, "issues": []}),
    ]
    validator = PromptValidator(client=mock_validator_client)

    mock_scorer_client = AsyncMock()
    mock_scorer_client.chat.return_value = make_mock_semantic_json(90, 90, 90, 90, 90)
    scorer = PromptScorer(client=mock_scorer_client)

    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=mock_analyzer,
        validator=validator,
        scorer=scorer,
    )

    result = await enhancer.enhance("5 ideas", validate=True)
    assert result.validation.is_valid is True
    assert result.quality is not None
    assert result.quality.overall_score >= 85
    # Scorer LLM was called exactly once (for the repaired version only)
    assert mock_scorer_client.chat.call_count == 1


# --- Call Budget Invariants (Normal 4 calls, Repair 6 calls) ---

@pytest.mark.asyncio
async def test_phase7_call_budget_invariants():
    # Scenario 1: First-pass valid clear request = 1 analyzer + 1 enhancer + 1 validator + 1 scorer = 4 calls
    analysis_clear = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build CLI",
        explicit_requirements=["Build CLI"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = analysis_clear

    mock_enhancer_client = AsyncMock()
    mock_enhancer_client.chat.return_value = "Build CLI."

    mock_validator_client = AsyncMock()
    mock_validator_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_validator_client)

    mock_scorer_client = AsyncMock()
    mock_scorer_client.chat.return_value = make_mock_semantic_json()
    scorer = PromptScorer(client=mock_scorer_client)

    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=mock_analyzer,
        validator=validator,
        scorer=scorer,
    )

    await enhancer.enhance("build CLI", validate=True)
    assert mock_analyzer.analyze.call_count == 1
    assert mock_enhancer_client.chat.call_count == 1
    assert mock_validator_client.chat.call_count == 1
    assert mock_scorer_client.chat.call_count == 1
    # Total = 1 + 1 + 1 + 1 = 4 LLM calls

    # Scenario 2: Repaired request = 1 analyzer + 2 enhancer + 2 validator + 1 scorer = 6 calls max
    mock_analyzer.analyze.reset_mock()
    mock_enhancer_client.chat.reset_mock()
    mock_validator_client.chat.reset_mock()
    mock_scorer_client.chat.reset_mock()

    mock_enhancer_client.chat.side_effect = ["Initial draft", "Repaired draft"]
    mock_validator_client.chat.side_effect = [
        json.dumps({"is_valid": False, "issues": [{"type": "MISSING_REQUIREMENT", "severity": "ERROR", "message": "m", "evidence": None}]}),
        json.dumps({"is_valid": True, "issues": []}),
    ]

    await enhancer.enhance("build CLI", validate=True)
    assert mock_analyzer.analyze.call_count == 1
    assert mock_enhancer_client.chat.call_count == 2
    assert mock_validator_client.chat.call_count == 2
    assert mock_scorer_client.chat.call_count == 1
    # Total = 1 + 2 + 2 + 1 = 6 LLM calls


# --- Metamorphic Tests ---

@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_a,prompt_b", [
    ("create a python script to parse csv and export json", "in python, make a script that parses csv files and outputs json"),
    ("explain the greenhouse effect simply in 3 bullet points", "using 3 simple bullet points, explain the greenhouse effect"),
])
async def test_phase7_metamorphic_scoring_consistency(prompt_a, prompt_b):
    mock_client = AsyncMock()
    mock_client.chat.return_value = make_mock_semantic_json(90, 90, 90, 90, 90)
    scorer = PromptScorer(client=mock_client)

    analysis_a = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Parse CSV and export JSON",
        explicit_requirements=["parse CSV", "export JSON"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    analysis_b = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Parse CSV and output JSON",
        explicit_requirements=["parse CSV", "output JSON"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    enh = "Act as a software developer. Create a Python script to parse CSV files and export JSON."
    val = PromptValidation(is_valid=True, issues=[])

    report_a = await scorer.score(prompt_a, analysis_a, enh, val)
    report_b = await scorer.score(prompt_b, analysis_b, enh, val)

    # Scores for equivalent intents must remain close within tolerance band (+/- 5 points)
    assert abs(report_a.overall_score - report_b.overall_score) <= 5


# ==============================================================================
# PHASE 3–7 PRODUCTION RELIABILITY CORRECTION: SOURCE GROUNDING INVARIANTS
# ==============================================================================

@pytest.mark.asyncio
async def test_reliability_group_a_unsupported_preference():
    """Test Group A: Mock analyzer returns a preference that does not exist in user input -> caught."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.9),
        objective="Draft a progress update",
        explicit_requirements=[],
        constraints=[],
        preferences=["dark theme and informal humor"],
        missing_context=[],
    )
    grounding = analyzer.check_analysis_grounding(analysis, "write a progress update for the team")
    assert not grounding.is_grounded
    assert any("dark theme" in item or "preference" in item for item in grounding.unsupported_items)


@pytest.mark.asyncio
async def test_reliability_group_b_unsupported_platform():
    """Test Group B: Original prompt does not state platform, analyzer invents one -> caught."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a cross-platform mobile calculator application",
        explicit_requirements=["cross-platform", "mobile"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    grounding = analyzer.check_analysis_grounding(analysis, "build me a calculator")
    assert not grounding.is_grounded
    assert any("mobile" in item or "cross-platform" in item for item in grounding.unsupported_items)


@pytest.mark.asyncio
async def test_reliability_group_c_category_drift():
    """Test Group C: Software creation request returned as SUMMARIZATION -> caught."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.SUMMARIZATION, confidence=0.95),
        objective="Summarize web tools",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    grounding = analyzer.check_analysis_grounding(analysis, "build a web tool for graphing equations")
    assert not grounding.is_grounded
    assert any("SUMMARIZATION" in item for item in grounding.unsupported_items)


@pytest.mark.asyncio
async def test_reliability_group_d_objective_drift():
    """Test Group D: Broad original prompt, analyzer objective invents specific audience/tech -> caught."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Create a cross-platform mobile application for college students with PostgreSQL",
        explicit_requirements=["PostgreSQL"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    grounding = analyzer.check_analysis_grounding(analysis, "create something")
    assert not grounding.is_grounded
    assert any("mobile" in item or "college students" in item or "postgresql" in item.lower() for item in grounding.unsupported_items)


@pytest.mark.asyncio
async def test_reliability_group_e_valid_analysis():
    """Test Group E: Paraphrased objective and genuinely supported requirements -> passes grounding."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a command-line Python tool to parse CSV and export JSON",
        explicit_requirements=["Python", "CLI", "CSV", "JSON"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    grounding = analyzer.check_analysis_grounding(analysis, "Create a CLI CSV parser in Python that outputs JSON")
    assert grounding.is_grounded
    assert len(grounding.unsupported_items) == 0


@pytest.mark.asyncio
async def test_reliability_group_f_clarification_authority():
    """Test Group F: Clarification answers add information not in original prompt -> authoritative user input."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build an iOS application for tracking personal expenses",
        explicit_requirements=["iOS", "expense tracking"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    answers = [
        ClarificationAnswer(question_id="q1", question="What platform?", answer="iOS"),
        ClarificationAnswer(question_id="q2", question="What purpose?", answer="expense tracking"),
    ]
    grounding = analyzer.check_analysis_grounding(
        analysis,
        raw_prompt="build an app",
        clarification_answers=answers,
    )
    assert grounding.is_grounded
    assert len(grounding.unsupported_items) == 0


@pytest.mark.asyncio
async def test_reliability_group_g_analysis_repair_success():
    """Test Group G: Attempt 1 has unsupported assertion; repair produces grounded result -> pipeline continues."""
    mock_client = AsyncMock()
    # Attempt 1: hallucinated React and mobile
    bad_output = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a mobile app with React",
        "explicit_requirements": ["React", "mobile"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {"score": 20, "level": "LOW", "needs_clarification": False, "reasons": [], "questions": []},
    })
    # Attempt 2 (repair): grounded analysis
    good_output = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a web quiz",
        "explicit_requirements": ["quiz"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["development platform or framework"],
        "ambiguity": {"score": 25, "level": "LOW", "needs_clarification": False, "reasons": [], "questions": []},
    })
    mock_client.chat.side_effect = [bad_output, good_output]

    analyzer = PromptAnalyzer(client=mock_client)
    res = await analyzer.analyze("make a quiz for my website")
    assert res.grounding.is_grounded
    assert "React" not in res.explicit_requirements
    assert mock_client.chat.call_count == 2


@pytest.mark.asyncio
async def test_reliability_group_h_analysis_repair_failure_stops_pipeline():
    """Test Group H: Analysis remains unsupported after repair -> pipeline stops cleanly (Enhancer=0, Validator=0, Scorer=0)."""
    mock_client = AsyncMock()
    bad_output = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build a mobile app with React for college students",
        "explicit_requirements": ["React", "mobile"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {"score": 20, "level": "LOW", "needs_clarification": False, "reasons": [], "questions": []},
    })
    # Both initial and repair return bad output
    mock_client.chat.side_effect = [bad_output, bad_output]

    mock_enhancer_client = AsyncMock()
    mock_validator_client = AsyncMock()
    mock_scorer_client = AsyncMock()

    analyzer = PromptAnalyzer(client=mock_client)
    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=analyzer,
        validator=PromptValidator(client=mock_validator_client),
        scorer=PromptScorer(client=mock_scorer_client),
    )

    result = await enhancer.enhance("make something")
    assert not result.analysis.grounding.is_grounded
    assert result.enhanced_prompt == ""
    assert result.validation is None
    assert result.quality is None
    # Strictly zero downstream calls
    assert mock_enhancer_client.chat.call_count == 0
    assert mock_validator_client.chat.call_count == 0
    assert mock_scorer_client.chat.call_count == 0


@pytest.mark.asyncio
async def test_reliability_group_i_validator_source_authority():
    """Test Group I: Analysis and enhancement agree with each other on B, but user asked A -> VALIDATION FAILURE."""
    validator = PromptValidator()
    # User asked A (email about sprint demo)
    user_prompt = "Write an email to my engineering team regarding tomorrow's sprint demo."
    # Analysis hallucinated B (CSV parser)
    hallucinated_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a Python CLI script to parse CSV files and export JSON",
        explicit_requirements=["Python", "CLI", "CSV", "JSON"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    # Enhancement perfectly implements B (CSV parser)
    enhancement_b = "Act as an experienced developer. Create a Python script to parse CSV files and export JSON."

    val = await validator.validate(
        original_prompt=user_prompt,
        analysis=hallucinated_analysis,
        enhanced_prompt=enhancement_b,
    )
    # MUST FAIL: mutual hallucination is rejected because original prompt is authoritative
    assert not val.is_valid
    assert any(
        i.type in {ValidationIssueType.ANALYSIS_OBJECTIVE_DRIFT, ValidationIssueType.OBJECTIVE_DRIFT}
        for i in val.issues
    )


@pytest.mark.asyncio
async def test_reliability_group_j_scoring_block():
    """Test Group J: Unsupported analysis must never receive a quality report."""
    mock_scorer_client = AsyncMock()
    scorer = PromptScorer(client=mock_scorer_client)

    ungrounded_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Invented goal",
        explicit_requirements=["React"],
        constraints=[],
        preferences=[],
        missing_context=[],
        grounding=AnalysisGrounding(is_grounded=False, unsupported_items=["React"]),
    )
    val = PromptValidation(is_valid=True, issues=[])

    with pytest.raises(ValueError, match="ungrounded"):
        await scorer.score("build something", ungrounded_analysis, "some prompt", val)
    assert mock_scorer_client.chat.call_count == 0


@pytest.mark.asyncio
async def test_reliability_group_k_ambiguity_integrity():
    """Test Group K: Analyzer cannot artificially lower ambiguity by hallucinating missing fields."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build an application",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["purpose and deliverable"],
        ambiguity=AmbiguityAssessment(score=10, level=AmbiguityLevel.LOW, needs_clarification=False),
    )
    # With zero grounded explicit requirements for ultra-vague request, ambiguity must remain HIGH
    analyzer._normalize_ambiguity(analysis, "make an app")
    assert analysis.ambiguity.score >= 75
    assert analysis.ambiguity.needs_clarification


@pytest.mark.asyncio
async def test_reliability_group_l_paraphrase_tolerance():
    """Test Group L: Valid semantic paraphrases pass without false rejection."""
    analyzer = PromptAnalyzer()
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a software tool to monitor CPU usage",
        explicit_requirements=["monitor CPU usage"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    grounding = analyzer.check_analysis_grounding(analysis, "develop a utility to monitor CPU usage")
    assert grounding.is_grounded
    assert len(grounding.unsupported_items) == 0


@pytest.mark.asyncio
async def test_reliability_task20_adversarial_self_consistency_synthetic():
    """Task 20: Synthetic adversarial self-consistency test.
    User asks A ("write a resignation letter").
    Analysis asserts B ("build a React calculator").
    Enhancement implements B ("Act as a developer. Build a React calculator...").
    System MUST reject it!
    """
    validator = PromptValidator()
    user_prompt = "Draft a formal resignation letter to my manager with two weeks notice."
    analysis_b = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a React calculator web application",
        explicit_requirements=["React", "calculator"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )
    enhanced_b = "Act as an experienced developer. Build a React calculator web application with clear state management."

    val = await validator.validate(
        original_prompt=user_prompt,
        analysis=analysis_b,
        enhanced_prompt=enhanced_b,
    )
    assert not val.is_valid
    drift_issues = [
        i for i in val.issues
        if i.type in {
            ValidationIssueType.ANALYSIS_OBJECTIVE_DRIFT,
            ValidationIssueType.OBJECTIVE_DRIFT,
            ValidationIssueType.UNSUPPORTED_ANALYSIS_REQUIREMENT,
        }
    ]
    assert len(drift_issues) > 0


@pytest.mark.asyncio
async def test_reliability_task24_call_budget_invariants():
    """Task 24: Call budget invariants.
    Normal clear grounded path: exactly 4 LLM calls (1 analyzer + 1 enhancer + 1 validator + 1 scorer).
    Analysis repair failure path: 2 analysis calls, 0 enhance, 0 validate, 0 scorer.
    """
    mock_analyzer = AsyncMock()
    mock_analyzer.check_analysis_grounding.return_value = AnalysisGrounding(is_grounded=True, evidence=[])
    mock_enhancer_client = AsyncMock()
    mock_validator_client = AsyncMock()
    mock_scorer_client = AsyncMock()

    grounded_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a Python tool",
        explicit_requirements=["Python"],
        constraints=[],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(score=10, level=AmbiguityLevel.LOW, needs_clarification=False),
        grounding=AnalysisGrounding(is_grounded=True, evidence=[]),
    )
    mock_analyzer.analyze.return_value = grounded_analysis

    mock_enhancer_client.chat.return_value = "Enhanced prompt text"
    mock_validator_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    mock_scorer_client.chat.return_value = json.dumps({
        "clarity": {"score": 90, "reason": "Clear"},
        "specificity": {"score": 90, "reason": "Specific"},
        "output_definition": {"score": 90, "reason": "Well defined"},
        "structure": {"score": 90, "reason": "Organized"},
        "concision": {"score": 90, "reason": "Concise"},
    })

    enhancer = PromptEnhancer(
        client=mock_enhancer_client,
        analyzer=mock_analyzer,
        validator=PromptValidator(client=mock_validator_client),
        scorer=PromptScorer(client=mock_scorer_client),
    )

    # 1. Normal path
    res = await enhancer.enhance("create a python script to parse logs", validate=True)
    assert res.quality is not None
    assert mock_analyzer.analyze.call_count == 1
    assert mock_enhancer_client.chat.call_count == 1
    assert mock_validator_client.chat.call_count == 1
    assert mock_scorer_client.chat.call_count == 1
    # Total = 1 + 1 + 1 + 1 = 4 LLM calls


# =====================================================================
# FINAL SEMANTIC GROUNDING HARDENING TESTS (TASKS 15 - 21)
# =====================================================================

@pytest.mark.asyncio
async def test_semantic_task15_contextual_modifier_attachment():
    """Task 15: Modifiers must attach to their contextual target, not a blind canonical preference.
    1. artifact complexity: "build a simple script" -> does NOT extract "simple explanation"
    2. communication tone: "write a professional email to my supervisor" -> extracts "professional tone"
    3. technical difficulty: "formal verification tool" -> does NOT extract "formal tone"
    4. output length: "write a brief summary" -> extracts "concise style"
    5. resource footprint: "use a light database" -> does NOT extract "light theme" or "light mode"
    """
    analyzer = PromptAnalyzer()

    # 1. Artifact complexity: "simple" modifies script, not explanation
    prefs_script = analyzer._recover_style_preferences("build a simple script to sort files", [])
    assert "simple explanation" not in prefs_script

    # 2. Communication tone: "professional" modifies email tone
    prefs_email = analyzer._recover_style_preferences("write a professional email to my supervisor", [])
    assert "professional tone" in prefs_email

    # 3. Technical difficulty: "formal" modifies verification tool, not writing tone
    prefs_formal = analyzer._recover_style_preferences("create a formal verification tool for smart contracts", [])
    assert "formal tone" not in prefs_formal

    # 4. Output length: "brief" modifies summary
    prefs_concise = analyzer._recover_style_preferences("write a brief summary of the meeting", [])
    assert "concise style" in prefs_concise

    # 5. Resource footprint: "light" modifies database
    prefs_light = analyzer._recover_style_preferences("use a light database for embedded devices", ["light theme"])
    assert "light theme" not in prefs_light


@pytest.mark.asyncio
async def test_semantic_task16_contrastive_pairs():
    """Task 16: Paired contrastive prompts where small wording changes alter modifier semantics.
    Pair 1: "give a simple explanation of recursion" vs "build a simple tool for file sorting"
    Pair 2: "write a professional letter to the board" vs "create a professional networking site"
    Pair 3: "develop a formal proof for theorem 1" vs "write a formal letter of complaint"
    Pair 4: "use a light database for caching" vs "design a dashboard with a light theme"
    Pair 5: "implement a fast search algorithm" vs "provide a fast response style"
    """
    analyzer = PromptAnalyzer()

    # --- Pair 1: simple explanation vs simple tool ---
    analysis_exp = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.EDUCATION, confidence=0.9),
        objective="Explain recursion",
        explicit_requirements=[],
        constraints=[],
        preferences=["simple explanation"],
        missing_context=[],
    )
    g_exp = analyzer.check_analysis_grounding(analysis_exp, "give a simple explanation of recursion")
    assert g_exp.is_grounded is True
    assert any(e.relation == GroundingRelation.MODIFIES_OUTPUT_STYLE for e in g_exp.evidence)

    analysis_tool = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build a file sorting tool",
        explicit_requirements=[],
        constraints=[],
        preferences=["simple explanation"],
        missing_context=[],
    )
    g_tool = analyzer.check_analysis_grounding(analysis_tool, "build a simple tool for file sorting")
    assert g_tool.is_grounded is False
    assert any("semantic mismatch: 'simple' modifies artifact deliverable" in u for u in g_tool.unsupported_items)

    # --- Pair 2: professional letter vs professional networking site ---
    analysis_prof_tone = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.9),
        objective="Write a letter to the board",
        explicit_requirements=[],
        constraints=[],
        preferences=["professional tone"],
        missing_context=[],
    )
    g_prof_tone = analyzer.check_analysis_grounding(analysis_prof_tone, "write a professional letter to the board")
    assert g_prof_tone.is_grounded is True

    analysis_prof_site = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Create a networking website",
        explicit_requirements=[],
        constraints=[],
        preferences=["professional tone"],
        missing_context=[],
    )
    g_prof_site = analyzer.check_analysis_grounding(analysis_prof_site, "create a professional networking site")
    assert g_prof_site.is_grounded is False
    assert any("semantic mismatch: 'professional' modifies artifact" in u for u in g_prof_site.unsupported_items)

    # --- Pair 3: formal proof vs formal letter ---
    analysis_formal_proof = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.RESEARCH, confidence=0.9),
        objective="Develop proof",
        explicit_requirements=[],
        constraints=[],
        preferences=["formal tone"],
        missing_context=[],
    )
    g_formal_proof = analyzer.check_analysis_grounding(analysis_formal_proof, "develop a formal proof for theorem 1")
    assert g_formal_proof.is_grounded is False
    assert any("semantic mismatch: 'formal' modifies method" in u for u in g_formal_proof.unsupported_items)

    analysis_formal_letter = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.9),
        objective="Write a formal letter of complaint",
        explicit_requirements=[],
        constraints=[],
        preferences=["formal tone"],
        missing_context=[],
    )
    g_formal_letter = analyzer.check_analysis_grounding(analysis_formal_letter, "write a formal letter of complaint")
    assert g_formal_letter.is_grounded is True

    # --- Pair 4: light database vs light theme ---
    analysis_light_db = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Use database",
        explicit_requirements=[],
        constraints=[],
        preferences=["light theme"],
        missing_context=[],
    )
    g_light_db = analyzer.check_analysis_grounding(analysis_light_db, "use a light database for caching")
    assert g_light_db.is_grounded is False
    assert any("semantic mismatch: 'light' modifies resource footprint" in u for u in g_light_db.unsupported_items)

    # --- Pair 5: fast algorithm vs fast response style ---
    analysis_fast_algo = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Search algorithm",
        explicit_requirements=[],
        constraints=[],
        preferences=["fast response style"],
        missing_context=[],
    )
    g_fast_algo = analyzer.check_analysis_grounding(analysis_fast_algo, "implement a fast search algorithm")
    assert g_fast_algo.is_grounded is False
    assert any("semantic mismatch: 'fast' modifies performance" in u for u in g_fast_algo.unsupported_items)


@pytest.mark.parametrize(
    "sparse_artifact_prompt,expected_deliverable",
    [
        ("build a tool", "tool"),
        ("write a document", "document"),
        ("make a plan", "plan"),
        ("create a presentation", "presentation"),
        ("build me a simple web app", "web app"),
    ],
)
@pytest.mark.asyncio
async def test_semantic_task17_sparse_purpose_triggers_clarification(sparse_artifact_prompt, expected_deliverable):
    """Task 17: Requests where artifact type is known but functional purpose is absent must trigger clarification."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "GENERAL", "confidence": 0.5},
        "objective": f"Create {expected_deliverable}",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": [],
        "missing_context": ["functional purpose", "core features"],
        "ambiguity": {
            "score": 75,
            "level": "HIGH",
            "needs_clarification": True,
            "reasons": ["Functional purpose is absent"],
            "questions": [
                {
                    "id": "q1",
                    "question": f"What is the primary purpose or functionality of the requested {expected_deliverable}?",
                    "reason": "Deliverable type is known but functional purpose is unstated.",
                    "target_dimension": "functional purpose",
                }
            ],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res = await analyzer.analyze(sparse_artifact_prompt)

    assert res.ambiguity.needs_clarification is True
    assert res.ambiguity.score >= 65
    assert res.ambiguity.level == AmbiguityLevel.HIGH
    assert len(res.ambiguity.questions) >= 1
    assert "functional purpose" in res.ambiguity.questions[0].target_dimension.lower()


@pytest.mark.parametrize(
    "well_specified_prompt",
    [
        "build a tool to count words in a text file",
        "write a document outlining the remote work policy",
        "make a plan for a 3-day trip to Tokyo with budget under 1000 dollars",
        "create a presentation about machine learning fundamentals for high school students",
    ],
)
@pytest.mark.asyncio
async def test_semantic_task18_well_specified_purpose_bypasses_clarification(well_specified_prompt):
    """Task 18: When artifact + purpose are provided, system must NOT trigger unnecessary clarification."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "GENERAL", "confidence": 0.9},
        "objective": "Execute specified task",
        "explicit_requirements": ["core requirements"],
        "constraints": [],
        "preferences": [],
        "missing_context": [],
        "ambiguity": {
            "score": 15,
            "level": "LOW",
            "needs_clarification": False,
            "reasons": [],
            "questions": [],
        },
    })

    analyzer = PromptAnalyzer(client=mock_client)
    res = await analyzer.analyze(well_specified_prompt)

    assert res.ambiguity.needs_clarification is False
    assert res.ambiguity.score < 30
    assert len(res.ambiguity.questions) == 0


@pytest.mark.asyncio
async def test_semantic_task19_adversarial_token_overlap_grounding_failure():
    """Task 19: Synthetic adversarial test where every token in an invented preference exists in user prompt,
    but the semantic relation is wrong. Grounding check MUST FAIL and validator must detect SEMANTIC_GROUNDING_MISMATCH.
    """
    analyzer = PromptAnalyzer()
    raw_prompt = "build a simple web app for professional networking with a fast algorithm"

    # All tokens ("simple", "professional", "fast") exist in raw_prompt,
    # but assigning them to "simple explanation", "professional tone", "fast response style" is WRONG.
    synthetic_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build a web application for professional networking with a fast algorithm",
        explicit_requirements=["networking"],
        constraints=[],
        preferences=["simple explanation", "professional tone", "fast response style"],
        missing_context=[],
        ambiguity=AmbiguityAssessment(score=10, level=AmbiguityLevel.LOW, needs_clarification=False),
    )

    grounding = analyzer.check_analysis_grounding(synthetic_analysis, raw_prompt)
    assert grounding.is_grounded is False
    assert len(grounding.unsupported_items) == 3
    assert any("simple" in u for u in grounding.unsupported_items)
    assert any("professional" in u for u in grounding.unsupported_items)
    assert any("fast" in u for u in grounding.unsupported_items)

    # Validator must also flag SEMANTIC_GROUNDING_MISMATCH deterministically
    validator = PromptValidator()
    issues = validator._check_deterministic_invariants(
        original_prompt=raw_prompt,
        analysis=synthetic_analysis,
        enhanced_prompt="Act as an expert developer. Build a simple web app.",
    )
    mismatch_issues = [i for i in issues if i.type == ValidationIssueType.SEMANTIC_GROUNDING_MISMATCH]
    assert len(mismatch_issues) >= 2


@pytest.mark.asyncio
async def test_semantic_task20_task21_quality_calibration_and_no_heading_bias():
    """Task 20 & 21: Quality calibration and heading bias invariant.
    - Well-specified prompt should have high specificity and output_definition.
    - Short but complete request without headings can score high (no heading/length penalty).
    """
    mock_scorer_client = AsyncMock()
    mock_scorer_client.chat.return_value = json.dumps({
        "clarity": {"score": 95, "reason": "Crystal clear instructions"},
        "specificity": {"score": 95, "reason": "Exact inputs and outputs specified"},
        "output_definition": {"score": 95, "reason": "Expected return value defined"},
        "structure": {"score": 95, "reason": "Clean 2-line structure perfect for single task"},
        "concision": {"score": 95, "reason": "Zero boilerplate filler"},
    })

    scorer = PromptScorer(client=mock_scorer_client)
    short_complete_prompt = "translate text from english to spanish"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.99),
        objective="Translate english text to spanish",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=[],
        grounding=AnalysisGrounding(is_grounded=True, evidence=[]),
    )
    # Clean 2-sentence enhanced prompt without markdown headings
    enhanced_prompt = "Translate the provided English text into Spanish accurately. Preserve the original tone and idioms without commentary."
    validation = PromptValidation(is_valid=True, issues=[])

    report = await scorer.score(
        original_prompt=short_complete_prompt,
        analysis=analysis,
        enhanced_prompt=enhanced_prompt,
        validation=validation,
    )

    # Complete prompt scores high despite being short and having no headings
    assert report.overall_score >= 90
    assert report.structure.score >= 90
    assert report.inflation_risk == InflationRisk.LOW


def _make_mock_analyzer():
    client = AsyncMock()
    async def chat_impl(messages, format=None):
        user_msg = messages[-1]["content"] if messages else ""
        # Return valid JSON representing basic extraction
        return json.dumps({
            "classification": {"category": "CODING", "confidence": 0.95},
            "objective": user_msg,
            "explicit_requirements": [],
            "constraints": [],
            "preferences": [],
            "missing_context": [],
            "ambiguity": {
                "score": 15,
                "level": "LOW",
                "needs_clarification": False,
                "reasons": [],
                "questions": [],
            },
        })
    client.chat.side_effect = chat_impl
    return PromptAnalyzer(client=client)


@pytest.mark.asyncio
async def test_functional_task19_contrastive_pairs():
    """Task 19: Generalized contrastive paired tests showing purpose-bearing versions have lower ambiguity.
    A. artifact only vs artifact + functional purpose
    B. artifact + platform vs artifact + platform + functional purpose
    C. artifact + style modifier vs artifact + actual purpose
    D. short complete operation vs short incomplete artifact request
    """
    analyzer = _make_mock_analyzer()

    # A. Artifact only vs Artifact + Functional Purpose
    incomplete_a = await analyzer.analyze("build a dashboard")
    assert incomplete_a.deliverable_semantics.is_functionally_complete is False
    assert incomplete_a.ambiguity.needs_clarification is True
    assert incomplete_a.ambiguity.score >= 65

    complete_a = await analyzer.analyze("build a dashboard for tracking warehouse inventory")
    assert complete_a.deliverable_semantics.is_functionally_complete is True
    assert complete_a.ambiguity.needs_clarification is False
    assert complete_a.ambiguity.score <= 30

    # B. Artifact + Platform vs Artifact + Platform + Purpose
    incomplete_b = await analyzer.analyze("make a telegram bot")
    assert incomplete_b.deliverable_semantics.is_functionally_complete is False
    assert incomplete_b.ambiguity.needs_clarification is True
    assert incomplete_b.ambiguity.score >= 65

    complete_b = await analyzer.analyze("make a telegram bot that answers customer support questions")
    assert complete_b.deliverable_semantics.is_functionally_complete is True
    assert complete_b.ambiguity.needs_clarification is False
    assert complete_b.ambiguity.score <= 30

    # C. Artifact + Style modifier vs Artifact + Actual purpose
    incomplete_c = await analyzer.analyze("write a professional email")
    assert incomplete_c.deliverable_semantics.is_functionally_complete is False
    assert incomplete_c.ambiguity.needs_clarification is True
    assert incomplete_c.ambiguity.score >= 65

    complete_c = await analyzer.analyze("write an email asking my lecturer for a deadline extension")
    assert complete_c.deliverable_semantics.is_functionally_complete is True
    assert complete_c.ambiguity.needs_clarification is False
    assert complete_c.ambiguity.score <= 30

    # D. Short complete operation vs Short incomplete artifact request
    complete_d = await analyzer.analyze("convert JSON to CSV using Python")
    assert complete_d.deliverable_semantics.is_functionally_complete is True
    assert complete_d.ambiguity.needs_clarification is False

    incomplete_d = await analyzer.analyze("make a tool")
    assert incomplete_d.deliverable_semantics.is_functionally_complete is False
    assert incomplete_d.ambiguity.needs_clarification is True


@pytest.mark.asyncio
async def test_functional_task20_cross_category():
    """Task 20: Test functional completeness across CODING, WRITING, PLANNING, CREATIVE, IMAGE_GENERATION, RESEARCH, EDUCATION."""
    analyzer = _make_mock_analyzer()

    pairs = [
        ("CODING", "build a web application", "build a web application to monitor server uptime"),
        ("WRITING", "draft an email", "draft an email announcing our product launch"),
        ("PLANNING", "make a roadmap", "make a roadmap for studying machine learning in 3 months"),
        ("CREATIVE", "write a story", "write a story about a time traveler in ancient Rome"),
        ("IMAGE_GENERATION", "create an image prompt", "create an image prompt of a cyberpunk city at night"),
        ("RESEARCH", "conduct research", "conduct research comparing SQL and NoSQL database performance"),
        ("EDUCATION", "create a curriculum", "create a curriculum to teach HTML CSS and JavaScript"),
    ]

    for category_name, incomplete_prompt, complete_prompt in pairs:
        res_incomplete = await analyzer.analyze(incomplete_prompt)
        assert res_incomplete.deliverable_semantics.is_functionally_complete is False, f"Failed for {category_name} incomplete: {incomplete_prompt}"
        assert res_incomplete.ambiguity.needs_clarification is True, f"Failed for {category_name} incomplete clarification: {incomplete_prompt}"

        res_complete = await analyzer.analyze(complete_prompt)
        assert res_complete.deliverable_semantics.is_functionally_complete is True, f"Failed for {category_name} complete: {complete_prompt}"
        assert res_complete.ambiguity.needs_clarification is False, f"Failed for {category_name} complete clarification: {complete_prompt}"


@pytest.mark.asyncio
async def test_functional_task21_adversarial_false_completeness():
    """Task 21: Synthetic prompt with platform, style, tone, tech adjectives, format, but NO actual function/purpose.
    Must NOT mistake descriptive richness for functional completeness.
    """
    analyzer = _make_mock_analyzer()
    prompt = "Develop an advanced, interactive, responsive, modern, dark-themed, lightweight Telegram bot in Python using asyncio."

    result = await analyzer.analyze(prompt)
    assert result.deliverable_semantics.is_functionally_complete is False
    assert result.ambiguity.needs_clarification is True
    assert result.ambiguity.score >= 65
    assert len(result.ambiguity.questions) >= 1
    assert "functional purpose" in result.ambiguity.questions[0].target_dimension.lower()


@pytest.mark.asyncio
async def test_functional_task22_short_complete_operations():
    """Task 22: Test very short but semantically complete requests do not trigger unnecessary clarification."""
    analyzer = _make_mock_analyzer()

    prompts = [
        "calculate the mortgage payment for 300000 loan at 5 percent",
        "summarize this article in 3 bullet points",
        "explain binary search simply",
        "sort this list of names alphabetically",
        "translate hello world into French",
    ]

    for p in prompts:
        res = await analyzer.analyze(p)
        assert res.deliverable_semantics.is_functionally_complete is True, f"Failed for complete prompt: {p}"
        assert res.ambiguity.needs_clarification is False, f"Clarification incorrectly triggered for complete prompt: {p}"
        assert res.ambiguity.score <= 30


@pytest.mark.asyncio
async def test_functional_task23_quality_relationship():
    """Task 23: Compare artifact-only request vs artifact + actual purpose.
    The purpose-bearing version receives higher output_definition and specificity.
    """
    scorer = PromptScorer(client=AsyncMock())

    # Prompt A: Artifact container only
    score_a = {
        "clarity": {"score": 85, "reason": "Sentence is clear"},
        "specificity": {"score": 60, "reason": "Lacks functional specifics"},
        "output_definition": {"score": 60, "reason": "Generic bot with no purpose defined"},
        "structure": {"score": 85, "reason": "Simple"},
        "concision": {"score": 90, "reason": "Short"},
    }
    # Prompt B: Artifact + Functional Purpose
    score_b = {
        "clarity": {"score": 95, "reason": "Very clear"},
        "specificity": {"score": 90, "reason": "Concrete functional tracking purpose"},
        "output_definition": {"score": 95, "reason": "Clear deliverable container and behavior"},
        "structure": {"score": 90, "reason": "Well structured"},
        "concision": {"score": 90, "reason": "Crisp"},
    }

    mock_client = AsyncMock()
    mock_client.chat.side_effect = [json.dumps(score_a), json.dumps(score_b)]
    scorer.client = mock_client

    analysis_a = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build a bot",
        deliverable_semantics=DeliverableSemantics(deliverable_type="bot", is_functionally_complete=False),
    )
    report_a = await scorer.score("build a bot", analysis_a, "Build a bot.", PromptValidation(is_valid=True))

    analysis_b = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a bot that tracks package deliveries",
        deliverable_semantics=DeliverableSemantics(deliverable_type="bot", intended_outcome="tracks package deliveries", is_functionally_complete=True),
    )
    report_b = await scorer.score("build a bot that tracks package deliveries", analysis_b, "Build a bot that tracks package deliveries.", PromptValidation(is_valid=True))

    assert report_b.specificity.score > report_a.specificity.score
    assert report_b.output_definition.score > report_a.output_definition.score
    assert report_b.overall_score > report_a.overall_score


@pytest.mark.asyncio
async def test_functional_task17_validator_catches_invented_purpose():
    """Task 17: Validator detects when an enhanced prompt invents a functional purpose that was unresolved in analysis."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validator = PromptValidator(client=mock_client)

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build a bot",
        deliverable_semantics=DeliverableSemantics(deliverable_type="bot", is_functionally_complete=False),
        ambiguity=AmbiguityAssessment(score=75, level=AmbiguityLevel.HIGH, needs_clarification=True),
    )

    # Enhanced prompt invents customer support e-commerce domain without clarification
    enhanced = "Act as an expert developer. Build an e-commerce customer support bot that manages orders and issues refunds."

    val_res = await validator.validate(
        original_prompt="build a bot",
        analysis=analysis,
        enhanced_prompt=enhanced,
        clarification_answers=[],
        allow_reasonable_assumptions=False,
    )

    assert val_res.is_valid is False
    assert any(
        issue.type in {ValidationIssueType.INVENTED_REQUIREMENT, ValidationIssueType.OBJECTIVE_DRIFT}
        for issue in val_res.issues
    )














