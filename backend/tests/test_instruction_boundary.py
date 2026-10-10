"""Tests for internal instruction and control-text boundary enforcement.

Verifies:
1. Contamination detection across all PromptAnalysis fields:
   - missing_context (Test A)
   - objective (Test B)
   - explicit_requirements (Test C)
   - preferences (Test D)
   - ambiguity.reasons (Test E)
   - ambiguity.questions (Test F)
2. Enhanced prompt meta-commentary / control-text detection (Test G)
3. Controlled 1-attempt analysis repair on contamination:
   - Successful repair allows normal pipeline continuation (Test H)
   - Failed repair halts pipeline with analysis_failed and ZERO downstream calls (Test I)
4. Source-based false-positive protection:
   - When user legitimately requests procedural wording ("Step 1", "Step 2"), it is allowed (Test J)
5. Adversarial multi-field injection detection
"""

import json
from unittest.mock import AsyncMock, patch
import pytest

from app.core.analyzer import PromptAnalyzer
from app.core.enhancer import PromptEnhancer
from app.core.instruction_boundary import (
    detect_control_text,
    scan_analysis_contamination,
    scan_enhanced_prompt_contamination,
)
from app.core.validator import PromptValidator
from app.schemas.analysis import (
    AmbiguityAssessment,
    AmbiguityLevel,
    ClarificationQuestion,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
)
from app.schemas.validation import ValidationIssueType, ValidationSeverity


@pytest.fixture
def analyzer():
    return PromptAnalyzer()


@pytest.fixture
def clean_base_analysis():
    return PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Create a Python CLI script to parse CSV files and export JSON",
        explicit_requirements=["Python", "CLI script", "parse CSV"],
        constraints=[],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(
            score=15,
            level=AmbiguityLevel.LOW,
            needs_clarification=False,
            reasons=[],
            questions=[],
        ),
    )


# ---------------------------------------------------------------------------
# TEST A — MISSING CONTEXT LEAK
# ---------------------------------------------------------------------------
def test_group_a_missing_context_leak(analyzer, clean_base_analysis):
    """Test A: Internal instruction heading inside missing_context triggers grounding/contamination failure."""
    raw_prompt = "Create a Python CLI script to parse CSV files and export JSON."
    clean_base_analysis.missing_context = ["STEP 4: assess ambiguity", "data storage method"]

    grounding = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert not grounding.is_grounded
    assert any("STEP 4" in item or "numbered step label" in item for item in grounding.unsupported_items)

    # Also test workflow procedure in missing_context
    clean_base_analysis.missing_context = ["perform source grounding"]
    grounding2 = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert not grounding2.is_grounded
    assert any("source grounding" in item for item in grounding2.unsupported_items)


# ---------------------------------------------------------------------------
# TEST B — OBJECTIVE LEAK
# ---------------------------------------------------------------------------
def test_group_b_objective_leak(analyzer, clean_base_analysis):
    """Test B: Objective containing system instruction language triggers grounding failure."""
    raw_prompt = "Build a simple weather dashboard."
    clean_base_analysis.objective = "STEP 2: EXTRACT STRUCTURED INFORMATION for weather dashboard"

    grounding = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert not grounding.is_grounded
    assert any("objective" in item and "STEP 2" in item for item in grounding.unsupported_items)


# ---------------------------------------------------------------------------
# TEST C — REQUIREMENT LEAK
# ---------------------------------------------------------------------------
def test_group_c_requirement_leak(analyzer, clean_base_analysis):
    """Test C: explicit_requirements containing schema or system instructions triggers grounding failure."""
    raw_prompt = "Write an essay about renewable energy."
    clean_base_analysis.classification = PromptClassification(category=PromptCategory.WRITING, confidence=0.95)
    clean_base_analysis.objective = "Write an essay on renewable energy"
    clean_base_analysis.explicit_requirements = ["renewable energy", "output must be valid JSON"]

    grounding = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert not grounding.is_grounded
    assert any("explicit requirement" in item and "valid json" in item.lower() for item in grounding.unsupported_items)


# ---------------------------------------------------------------------------
# TEST D — PREFERENCE LEAK
# ---------------------------------------------------------------------------
def test_group_d_preference_leak(analyzer, clean_base_analysis):
    """Test D: preferences containing rubric or system prompt language triggers grounding failure."""
    raw_prompt = "Explain quantum computing simply."
    clean_base_analysis.classification = PromptClassification(category=PromptCategory.EDUCATION, confidence=0.95)
    clean_base_analysis.objective = "Explain quantum computing simply"
    clean_base_analysis.preferences = ["clarity rubric", "simple explanation"]

    grounding = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert not grounding.is_grounded
    assert any("preference" in item and "clarity rubric" in item for item in grounding.unsupported_items)


# ---------------------------------------------------------------------------
# TEST E — AMBIGUITY-REASON LEAK
# ---------------------------------------------------------------------------
def test_group_e_ambiguity_reason_leak(analyzer, clean_base_analysis):
    """Test E: ambiguity.reasons including copied system prompt text triggers grounding failure."""
    raw_prompt = "Build an app."
    clean_base_analysis.ambiguity.score = 75
    clean_base_analysis.ambiguity.level = AmbiguityLevel.HIGH
    clean_base_analysis.ambiguity.needs_clarification = True
    clean_base_analysis.ambiguity.reasons = [
        "System instructions require high ambiguity assessment when missing platform and domain",
    ]

    grounding = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert not grounding.is_grounded
    assert any("ambiguity reason" in item and "system instructions" in item.lower() for item in grounding.unsupported_items)


# ---------------------------------------------------------------------------
# TEST F — CLARIFICATION QUESTION LEAK
# ---------------------------------------------------------------------------
def test_group_f_clarification_question_leak(analyzer, clean_base_analysis):
    """Test F: Generated question asking about internal analyzer workflow or schema is rejected."""
    raw_prompt = "Build an app."
    clean_base_analysis.ambiguity.score = 75
    clean_base_analysis.ambiguity.level = AmbiguityLevel.HIGH
    clean_base_analysis.ambiguity.needs_clarification = True
    clean_base_analysis.ambiguity.questions = [
        ClarificationQuestion(
            id="q1",
            question="Please provide the information required by the schema for STEP 4.",
            reason="Schema contract compliance",
            target_dimension="schema requirements",
        )
    ]

    grounding = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert not grounding.is_grounded
    assert any("clarification question" in item for item in grounding.unsupported_items)


# ---------------------------------------------------------------------------
# TEST G — ENHANCED PROMPT META LEAK
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_group_g_enhanced_prompt_meta_leak():
    """Test G: Enhanced prompt containing template meta-instructions or validator references is rejected."""
    raw_prompt = "Build a portfolio website in Vue."
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="Build a portfolio website in Vue",
        explicit_requirements=["portfolio website", "Vue"],
        constraints=[],
        preferences=[],
        missing_context=[],
    )

    contaminated_prompt = (
        "Act as an experienced frontend developer.\n\n"
        "OBJECTIVE: Build a personal portfolio website in Vue.\n\n"
        "CORE REQUIREMENTS:\n"
        "- Vue framework\n"
        "- Portfolio showcases\n\n"
        "According to the validator, ensure intent preservation score remains high.\n"
        "Follow category template guidance closely."
    )

    validator = PromptValidator()
    validation = await validator.validate(
        original_prompt=raw_prompt,
        analysis=analysis,
        enhanced_prompt=contaminated_prompt,
    )

    assert not validation.is_valid
    meta_issues = [i for i in validation.issues if i.type == ValidationIssueType.META_COMMENTARY]
    assert len(meta_issues) > 0
    assert any(
        "validator" in i.message.lower() or "template" in i.message.lower()
        for i in meta_issues
    )


# ---------------------------------------------------------------------------
# TEST H — REPAIR SUCCESS
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_group_h_repair_success(clean_base_analysis):
    """Test H: Initial structured analysis contains internal instructions. One repair removes it -> normal pipeline continues."""
    raw_prompt = "Create a Python CLI script to parse CSV files and export JSON."

    # Contaminated first attempt
    contaminated_first = clean_base_analysis.model_copy(deep=True)
    contaminated_first.missing_context = ["STEP 4: assess ambiguity"]

    # Clean second attempt
    clean_second = clean_base_analysis.model_copy(deep=True)
    clean_second.missing_context = ["file encoding"]

    mock_client = AsyncMock()
    # 1st chat call: analyzer attempt 1 (contaminated)
    # 2nd chat call: analyzer repair attempt (clean)
    # 3rd chat call: enhancer
    # 4th chat call: validator
    # 5th chat call: scorer
    mock_client.chat.side_effect = [
        contaminated_first.model_dump_json(),
        clean_second.model_dump_json(),
        "Act as a Python developer. Build a CLI script to parse CSV files and export JSON.\n\nCORE REQUIREMENTS:\n- Python\n- Parse CSV\n- Export JSON",
        json.dumps({"is_valid": True, "issues": []}),
        json.dumps({
            "clarity": {"score": 90, "reason": "Clear."},
            "specificity": {"score": 90, "reason": "Specific."},
            "output_definition": {"score": 90, "reason": "Defined."},
            "structure": {"score": 90, "reason": "Structured."},
            "concision": {"score": 90, "reason": "Concise."},
        }),
    ]

    analyzer = PromptAnalyzer(client=mock_client)
    validator = PromptValidator(client=mock_client)
    enhancer = PromptEnhancer(client=mock_client, analyzer=analyzer, validator=validator)

    res = await enhancer.enhance(raw_prompt)

    # Pipeline successfully completed
    assert res.analysis.grounding.is_grounded
    assert res.enhanced_prompt != ""
    assert res.validation.is_valid
    assert res.analysis.missing_context == ["file encoding"]


# ---------------------------------------------------------------------------
# TEST I — REPAIR FAILURE
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_group_i_repair_failure(clean_base_analysis):
    """Test I: Both initial and repaired analyses are contaminated -> stops pipeline, analysis_failed, zero downstream calls."""
    raw_prompt = "Create a Python CLI script to parse CSV files and export JSON."

    contaminated_analysis = clean_base_analysis.model_copy(deep=True)
    contaminated_analysis.missing_context = ["STEP 4: assess ambiguity"]

    mock_client = AsyncMock()
    # 1st call: initial analysis (contaminated)
    # 2nd call: repair analysis (still contaminated)
    mock_client.chat.side_effect = [
        contaminated_analysis.model_dump_json(),
        contaminated_analysis.model_dump_json(),
    ]

    analyzer = PromptAnalyzer(client=mock_client)
    validator = PromptValidator(client=mock_client)
    enhancer = PromptEnhancer(client=mock_client, analyzer=analyzer, validator=validator)

    res = await enhancer.enhance(raw_prompt)

    # Grounding failed, enhancement aborted
    assert not res.analysis.grounding.is_grounded
    assert res.enhanced_prompt == ""
    assert res.validation is None
    assert res.quality is None
    # Downstream models were NEVER called: total chat calls = 2 (initial analysis + repair analysis)
    assert mock_client.chat.call_count == 2


# ---------------------------------------------------------------------------
# TEST J — USER ACTUALLY REQUESTS SIMILAR WORDING
# ---------------------------------------------------------------------------
def test_group_j_user_requests_procedural_wording(analyzer, clean_base_analysis):
    """Test J: If USER genuinely writes 'Step 1...', 'Step 2...', those words are authoritative and NOT rejected."""
    raw_prompt = (
        "Write a technical tutorial explaining binary search with:\n"
        "Step 1: Problem definition\n"
        "Step 2: Algorithm logic\n"
        "Step 3: Complexity analysis"
    )

    clean_base_analysis.classification = PromptClassification(category=PromptCategory.EDUCATION, confidence=0.95)
    clean_base_analysis.objective = "Write a binary search tutorial with Step 1, Step 2, and Step 3"
    clean_base_analysis.explicit_requirements = [
        "Step 1: Problem definition",
        "Step 2: Algorithm logic",
        "Step 3: Complexity analysis",
    ]
    clean_base_analysis.missing_context = []

    grounding = analyzer.check_analysis_grounding(clean_base_analysis, raw_prompt)
    assert grounding.is_grounded, f"Expected grounded, got unsupported items: {grounding.unsupported_items}"


# ---------------------------------------------------------------------------
# ADVERSARIAL MULTI-FIELD INJECTION TEST
# ---------------------------------------------------------------------------
def test_adversarial_multi_field_contamination():
    """Adversarial test: analysis injects internal system phrases across multiple fields simultaneously."""
    raw_prompt = "Build a fitness tracking app."
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.95),
        objective="STEP 1: TASK CLASSIFICATION for fitness tracking",
        explicit_requirements=["fitness tracking", "adhere to required schema"],
        constraints=["rule of neutrality"],
        preferences=["confidence guidelines"],
        missing_context=["STEP 6: source grounding provenance"],
        ambiguity=AmbiguityAssessment(
            score=80,
            level=AmbiguityLevel.HIGH,
            needs_clarification=True,
            reasons=["system instructions say ambiguity is high"],
            questions=[
                ClarificationQuestion(
                    id="q1",
                    question="Which schema contract should be satisfied?",
                    reason="schema requirement",
                    target_dimension="json schema",
                )
            ],
        ),
    )

    issues = scan_analysis_contamination(analysis, raw_prompt)
    # Every single field should be detected!
    assert len(issues) >= 6
    assert any("objective" in i for i in issues)
    assert any("explicit requirement" in i for i in issues)
    assert any("constraint" in i for i in issues)
    assert any("preference" in i for i in issues)
    assert any("missing context" in i for i in issues)
    assert any("ambiguity reason" in i for i in issues)
    assert any("clarification question" in i for i in issues)
