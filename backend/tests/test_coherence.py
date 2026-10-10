"""
Final General Semantic-Coherence Hardening Test Suite (Tasks 21 - 27).
Validates cross-field semantic consistency across Phases 1-8 architecture:
- Task 21: Regression cases for consumed diagnostic examples.
- Task 22: Generalization test set across all 11 action families.
- Task 23: Contrastive category tests (same topic, different actions).
- Task 24: Compound intent tests (deliverable preservation).
- Task 25: Missing context task-coherence tests.
- Task 26: Interrogative question vs neutral dimension separation.
- Task 27: Adversarial internal task-family consistency (mutual hallucination defense).
"""

import json
from unittest.mock import AsyncMock
import pytest

from app.core.analyzer import PromptAnalyzer
from app.core.intent_anchor import (
    ACTION_FAMILY_TO_CATEGORY,
    check_analysis_coherence,
    extract_intent_anchor,
    normalize_missing_context_dimension,
)
from app.core.validator import PromptValidator
from app.schemas.analysis import (
    ActionFamily,
    AmbiguityAssessment,
    AmbiguityLevel,
    AnalysisGrounding,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
)
from app.schemas.validation import ValidationIssueType, ValidationSeverity


# ==============================================================================
# TASK 21: REGRESSION CASES (CONSUMED DIAGNOSTIC EXAMPLES)
# Asserts semantic properties without hardcoded noun branching in production.
# ==============================================================================

@pytest.mark.asyncio
async def test_task21_regression_software_creation_not_data_analysis():
    """Diagnostic 1: Software implementation request must be CODING, not DATA_ANALYSIS."""
    prompt = "Create a Go function to parse timestamps and calculate time delta"
    anchor = extract_intent_anchor(prompt)
    assert anchor.action_family == ActionFamily.SOFTWARE_CREATION
    assert anchor.is_strong is True

    mock_client = AsyncMock()
    # Model stochastically misclassified as DATA_ANALYSIS due to timestamps/delta vocabulary
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "DATA_ANALYSIS", "confidence": 0.85},
        "objective": "Create a function to parse timestamps and compute delta",
        "explicit_requirements": ["Go", "parse timestamps", "calculate time delta"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["expected timestamp string format"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    # Classification consistency gate must align to CODING
    assert result.classification.category == PromptCategory.CODING
    assert "timestamp format" in " ".join(result.missing_context).lower() or len(result.missing_context) >= 1


@pytest.mark.asyncio
async def test_task21_regression_writing_not_debugging():
    """Diagnostic 2: Composition request must be WRITING, not DEBUGGING."""
    prompt = "Draft a polite message to a vendor about missing an invoice deadline"
    anchor = extract_intent_anchor(prompt)
    assert anchor.action_family == ActionFamily.WRITING
    assert anchor.is_strong is True

    mock_client = AsyncMock()
    # Model stochastically misclassified as DEBUGGING due to problem/deadline vocabulary
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "DEBUGGING", "confidence": 0.80},
        "objective": "Draft a polite message to a vendor regarding invoice deadline",
        "explicit_requirements": [],
        "constraints": [],
        "preferences": ["polite tone"],
        "missing_context": ["vendor name", "invoice number"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    assert result.classification.category == PromptCategory.WRITING
    assert not any(tech in " ".join(result.missing_context).lower() for tech in ["platform", "framework", "database"])


@pytest.mark.asyncio
async def test_task21_regression_conceptual_explanation_not_general():
    """Diagnostic 3: Conceptual explanation question must be EDUCATION, not GENERAL."""
    prompt = "What is the difference between TCP and UDP?"
    anchor = extract_intent_anchor(prompt)
    assert anchor.action_family == ActionFamily.EDUCATION_EXPLANATION
    assert anchor.is_strong is True

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "GENERAL", "confidence": 0.60},
        "objective": "Explain the differences between TCP and UDP",
        "explicit_requirements": ["compare TCP and UDP"],
        "constraints": [],
        "preferences": [],
        "missing_context": ["target technical depth"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    assert result.classification.category == PromptCategory.EDUCATION


# ==============================================================================
# TASK 22: GENERALIZATION TEST SET ACROSS ALL 11 ACTION FAMILIES
# Uses entirely new, unrelated domains to prove general applicability.
# ==============================================================================

@pytest.mark.parametrize(
    "prompt,expected_family,expected_category",
    [
        # 1. Software Creation: Rust command-line conversion utility
        (
            "Write a Rust command-line utility to convert YAML configuration files into TOML format",
            ActionFamily.SOFTWARE_CREATION,
            PromptCategory.CODING,
        ),
        # 2. Software Debugging: SQL concurrency deadlock diagnosis
        (
            "Why does my SQL query deadlock when concurrent bulk inserts occur in production?",
            ActionFamily.SOFTWARE_DEBUGGING,
            PromptCategory.DEBUGGING,
        ),
        # 3. Writing: Formal condolence letter
        (
            "Compose a formal condolence letter on behalf of our team to a bereaved colleague",
            ActionFamily.WRITING,
            PromptCategory.WRITING,
        ),
        # 4. Education / Explanation: Photosynthesis biological mechanism
        (
            "Explain how photosynthesis converts sunlight and carbon dioxide into glucose",
            ActionFamily.EDUCATION_EXPLANATION,
            PromptCategory.EDUCATION,
        ),
        # 5. Research: Clinical trial evidence on mRNA vaccines
        (
            "Investigate and synthesize clinical trial evidence on mRNA vaccine longevity and efficacy",
            ActionFamily.RESEARCH,
            PromptCategory.RESEARCH,
        ),
        # 6. Planning: Beginner marathon 6-month itinerary
        (
            "Create a 6-month training itinerary for a beginner preparing for their first marathon",
            ActionFamily.PLANNING,
            PromptCategory.PLANNING,
        ),
        # 7. Data Analysis: Monthly electricity consumption readings
        (
            "Analyze this dataset of monthly electricity consumption readings to identify anomalies and peak load hours",
            ActionFamily.DATA_ANALYSIS,
            PromptCategory.DATA_ANALYSIS,
        ),
        # 8. Creative Ideation: Brand names for bamboo toothbrushes
        (
            "Brainstorm 10 catchy brand names for an eco-friendly bamboo toothbrush line",
            ActionFamily.CREATIVE_IDEATION,
            PromptCategory.CREATIVE,
        ),
        # 9. Image Generation: Mars greenhouse digital painting
        (
            "Generate a vibrant digital illustration of an astronaut tending a greenhouse on Mars",
            ActionFamily.IMAGE_CREATION,
            PromptCategory.IMAGE_GENERATION,
        ),
        # 10. Summarization: Municipal zoning document
        (
            "Summarize this 10-page municipal zoning report into key takeaways for local homeowners",
            ActionFamily.SUMMARIZATION,
            PromptCategory.SUMMARIZATION,
        ),
        # 11. General Fallback: Broad open-ended philosophical inquiry
        (
            "Share your thoughts and general perspectives on the long-term impact of remote work on urbanization",
            ActionFamily.GENERAL,
            PromptCategory.GENERAL,
        ),
    ],
)
def test_task22_generalization_all_eleven_action_families(prompt, expected_family, expected_category):
    """Generalization: Verify action family and category alignment across 11 unrelated domains."""
    anchor = extract_intent_anchor(prompt)
    assert anchor.action_family == expected_family
    if expected_family != ActionFamily.GENERAL:
        assert anchor.is_strong is True
        assert ACTION_FAMILY_TO_CATEGORY[anchor.action_family] == expected_category
    else:
        assert anchor.is_strong is False


# ==============================================================================
# TASK 23: CONTRASTIVE CATEGORY TESTS (SAME TOPIC, DIFFERENT ACTIONS)
# Proves classification strictly follows the requested ACTION, not subject vocabulary.
# ==============================================================================

@pytest.mark.parametrize(
    "topic_name,prompt_a,expected_cat_a,prompt_b,expected_cat_b",
    [
        # Contrastive Pair 1: Microservices
        (
            "microservices",
            "Build a Python microservice to process incoming payments",
            PromptCategory.CODING,
            "Explain how microservices communicate via asynchronous message brokers",
            PromptCategory.EDUCATION,
        ),
        # Contrastive Pair 2: Compiler optimization
        (
            "compiler optimization",
            "Debug this compiler optimization pass that fails with an assertion crash",
            PromptCategory.DEBUGGING,
            "Research recent academic publications on LLVM compiler optimization passes",
            PromptCategory.RESEARCH,
        ),
        # Contrastive Pair 3: Remote work travel policy
        (
            "travel policy",
            "Draft an announcement email to employees about the new travel expense policy",
            PromptCategory.WRITING,
            "Summarize this 20-page travel expense policy into a one-page summary",
            PromptCategory.SUMMARIZATION,
        ),
        # Contrastive Pair 4: Financial portfolio
        (
            "investment portfolio",
            "Analyze this historical portfolio returns table to compute volatility metrics",
            PromptCategory.DATA_ANALYSIS,
            "Plan a 5-year investment portfolio roadmap for a young professional",
            PromptCategory.PLANNING,
        ),
    ],
)
def test_task23_contrastive_category_pairs(topic_name, prompt_a, expected_cat_a, prompt_b, expected_cat_b):
    """Contrastive: Verify identical topic nouns yield different categories when actions differ."""
    anchor_a = extract_intent_anchor(prompt_a)
    anchor_b = extract_intent_anchor(prompt_b)

    cat_a = ACTION_FAMILY_TO_CATEGORY.get(anchor_a.action_family, PromptCategory.GENERAL)
    cat_b = ACTION_FAMILY_TO_CATEGORY.get(anchor_b.action_family, PromptCategory.GENERAL)

    assert cat_a == expected_cat_a
    assert cat_b == expected_cat_b
    assert cat_a != cat_b, f"Topic '{topic_name}' failed to differentiate action semantics!"


# ==============================================================================
# TASK 24: COMPOUND INTENT TESTS (DELIVERABLE PRESERVATION)
# Multi-action requests retain primary category while preserving secondary deliverables.
# ==============================================================================

@pytest.mark.asyncio
async def test_task24_compound_intent_preserves_all_deliverables():
    """Compound: Verify dominant category is selected and all explicit secondary deliverables survive."""
    prompt = "Research thermal battery storage technologies, compare their efficiency, and draft a summary memorandum for executive leadership"

    mock_client = AsyncMock()
    mock_client.chat.return_value = json.dumps({
        "classification": {"category": "RESEARCH", "confidence": 0.90},
        "objective": "Research thermal battery storage technologies, compare efficiency, and draft executive memorandum",
        "explicit_requirements": [
            "research thermal battery storage",
            "compare efficiency",
            "executive summary memorandum",
        ],
        "constraints": [],
        "preferences": [],
        "missing_context": ["temperature range specifications"],
    })

    analyzer = PromptAnalyzer(client=mock_client)
    result = await analyzer.analyze(prompt)

    # Primary category is RESEARCH
    assert result.classification.category == PromptCategory.RESEARCH
    # All secondary deliverables survive in explicit_requirements
    reqs_str = " ".join(result.explicit_requirements).lower()
    assert "compare" in reqs_str
    assert "memorandum" in reqs_str or "summary" in reqs_str


# ==============================================================================
# TASK 25: MISSING CONTEXT COHERENCE TESTS
# Detects and rejects irrelevant cross-domain dimensions, while preserving valid ones.
# ==============================================================================

def test_task25_incoherent_missing_dimension_rejected():
    """Missing Context: Technical software dimensions are rejected on pure writing tasks."""
    dim = "development platform or framework"
    action_family = ActionFamily.WRITING
    prompt = "Draft an apology letter to a customer regarding a delayed shipment"

    normalized = normalize_missing_context_dimension(dim, action_family, prompt)
    assert normalized is None, "Incoherent software framework was not rejected for a writing task!"


def test_task25_valid_relevant_dimension_survives():
    """Missing Context: Relevant domain dimensions survive normalization."""
    dim = "target audience expertise level"
    action_family = ActionFamily.EDUCATION_EXPLANATION
    prompt = "Explain how neural networks learn via backpropagation"

    normalized = normalize_missing_context_dimension(dim, action_family, prompt)
    assert normalized == "target audience expertise level"


# ==============================================================================
# TASK 26: QUESTION VS DIMENSION SEPARATION TESTS
# Interrogative sentences in missing_context are safely normalized or rejected.
# ==============================================================================

def test_task26_interrogative_normalized_to_neutral_dimension():
    """Separation: Clarification question is converted to neutral dimension name."""
    question_dim = "What specific aspects of the product rollout need planning?"
    action_family = ActionFamily.PLANNING
    prompt = "Plan a comprehensive product rollout schedule for next quarter"

    normalized = normalize_missing_context_dimension(question_dim, action_family, prompt)
    assert normalized is not None
    assert not normalized.endswith("?")
    assert "rollout scope" in normalized.lower() or "scope" in normalized.lower()


def test_task26_unresolvable_interrogative_rejected():
    """Separation: Vague unresolvable question in missing context is rejected."""
    question_dim = "Why should we do this?"
    action_family = ActionFamily.PLANNING
    prompt = "Plan a marketing rollout"

    normalized = normalize_missing_context_dimension(question_dim, action_family, prompt)
    assert normalized is None


# ==============================================================================
# TASK 27: ADVERSARIAL INTERNAL CONSISTENCY TEST (TASK-FAMILY MUTUAL HALLUCINATION)
# Derived fields agree internally on Task B, but authoritative input was Task A.
# Coherence gate MUST catch and reject the analysis.
# ==============================================================================

def test_task27_adversarial_task_family_mutual_hallucination_rejected():
    """Adversarial: When derived fields agree on CODING but user requested WRITING, coherence gate rejects it."""
    authoritative_user_input = "Draft a polite follow-up message to the conference organizer asking about speaker guidelines"
    anchor = extract_intent_anchor(authoritative_user_input)
    assert anchor.action_family == ActionFamily.WRITING

    # Malicious synthetic analysis: internally consistent with CODING, but contradicts authoritative request
    adversarial_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Implement a speaker guidelines conference management portal",
        explicit_requirements=["conference portal", "speaker guidelines module"],
        constraints=[],
        preferences=[],
        missing_context=["development platform or framework"],
        ambiguity=AmbiguityAssessment(score=15, level=AmbiguityLevel.LOW, needs_clarification=False),
        grounding=AnalysisGrounding(is_grounded=True, evidence=[]),
    )

    violations = check_analysis_coherence(authoritative_user_input, adversarial_analysis, anchor)

    assert len(violations) >= 1
    violations_text = " ".join(violations).lower()
    assert "classification" in violations_text or "objective" in violations_text
    assert "writing" in violations_text or "coding" in violations_text


# ==============================================================================
# TASK 18: VALIDATOR TASK-FAMILY DRIFT DEFENSE
# Validator must catch when enhancement drifts to an incompatible task family.
# ==============================================================================

@pytest.mark.asyncio
async def test_task18_validator_catches_task_family_drift():
    """Validator: Detects TASK_FAMILY_DRIFT when writing prompt is converted to code implementation."""
    user_prompt = "Draft a formal resignation letter to my manager"
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.WRITING, confidence=0.95),
        objective="Draft a formal resignation letter",
        explicit_requirements=[],
        constraints=[],
        preferences=["formal tone"],
        missing_context=["last working day"],
        ambiguity=AmbiguityAssessment(score=10, level=AmbiguityLevel.LOW, needs_clarification=False),
        grounding=AnalysisGrounding(is_grounded=True, evidence=[]),
    )

    # Incompatible enhanced prompt: instead of a letter, instructs building a software app
    drifted_enhanced_prompt = (
        "Act as an experienced full-stack engineer. Build an interactive web app with React "
        "and a database schema to manage employee resignations and HR tracking."
    )

    validator = PromptValidator()
    deterministic_issues = validator._check_deterministic_invariants(
        original_prompt=user_prompt,
        analysis=analysis,
        enhanced_prompt=drifted_enhanced_prompt,
    )

    drift_issues = [i for i in deterministic_issues if i.type == ValidationIssueType.TASK_FAMILY_DRIFT]
    assert len(drift_issues) >= 1
    assert drift_issues[0].severity == ValidationSeverity.ERROR
