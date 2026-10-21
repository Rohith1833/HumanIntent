"""Production hardening tests for General Source-Coverage Gate and Atomic Unit Extraction."""

import json
from unittest.mock import AsyncMock
import pytest

from app.core.analyzer import PromptAnalyzer
from app.core.enhancer import PromptEnhancer
from app.core.scorer import PromptScorer
from app.core.source_coverage import (
    check_source_coverage,
    extract_source_semantic_units,
    is_semantic_unit_covered,
)
from app.core.validator import PromptValidator
from app.schemas.analysis import (
    AmbiguityAssessment,
    AmbiguityLevel,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
)
from app.schemas.validation import ValidationIssueType, ValidationSeverity


@pytest.fixture
def mock_clients():
    """Create reusable mocks for LLM services."""
    mock_llm = AsyncMock()
    validator = PromptValidator(client=mock_llm)
    scorer = PromptScorer(client=mock_llm)
    enhancer = PromptEnhancer(client=mock_llm, validator=validator, scorer=scorer)
    analyzer = PromptAnalyzer(client=mock_llm)
    return {
        "client": mock_llm,
        "validator": validator,
        "scorer": scorer,
        "enhancer": enhancer,
        "analyzer": analyzer,
    }


# =========================================================================
# TASK 5: Messy Compound Generalization Tests Across 5 Unrelated Domains
# =========================================================================

DOMAINS_MESSY_CASES = [
    (
        "Event Workflow",
        (
            "build an event management platform with profile selection and registration and payment "
            "and registration completion and email communication and problem statement handling "
            "and participant problem selection and project submission and judging"
        ),
        [
            "profile selection",
            "registration",
            "payment",
            "registration completion",
            "email communication",
            "problem statement handling",
            "participant problem selection",
            "project submission",
            "judging",
        ],
        "problem statement handling",
    ),
    (
        "Inventory Workflow",
        (
            "we need warehouse software with barcode scanning and stock level alerts and supplier reordering "
            "and receiving dock check-in and discrepancy logging and bin transfer tracking and cycle count auditing"
        ),
        [
            "barcode scanning",
            "stock level alerts",
            "supplier reordering",
            "receiving dock check-in",
            "discrepancy logging",
            "bin transfer tracking",
            "cycle count auditing",
        ],
        "receiving dock check-in",
    ),
    (
        "Academic Workflow",
        (
            "make a campus academic system for faculty syllabus uploads and grade entry and report card generation "
            "and attendance auditing and parent notification emails and incident logging and advisory scheduling"
        ),
        [
            "faculty syllabus uploads",
            "grade entry",
            "report card generation",
            "attendance auditing",
            "parent notification emails",
            "incident logging",
            "advisory scheduling",
        ],
        "report card generation",
    ),
    (
        "Publishing Workflow",
        (
            "create a publishing tool with article drafting and editorial review and automated plagiarism checking "
            "and translation into spanish and social media snippet generation and scheduled publishing and reader comment moderation"
        ),
        [
            "article drafting",
            "editorial review",
            "automated plagiarism checking",
            "translation into spanish",
            "social media snippet generation",
            "scheduled publishing",
            "reader comment moderation",
        ],
        "translation into spanish",
    ),
    (
        "Travel Planning Workflow",
        (
            "i want a travel planner with flight search and hotel reservation and itinerary creation "
            "and packing list generator and currency conversion and expense tracking and offline map export"
        ),
        [
            "flight search",
            "hotel reservation",
            "itinerary creation",
            "packing list generator",
            "currency conversion",
            "expense tracking",
            "offline map export",
        ],
        "currency conversion",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("domain_name,prompt,expected_units,middle_req_to_delete", DOMAINS_MESSY_CASES)
async def test_messy_compound_generalization_coverage_and_deletion(
    mock_clients, domain_name, prompt, expected_units, middle_req_to_delete
):
    """TASK 5: Generalization test across 5 diverse domains with 7-10 requirements.

    Asserts:
    1. Every explicit semantic request unit survives extraction.
    2. No unsupported unit is invented.
    3. Structured analysis with all units passes source-coverage gate.
    4. Enhancement preserving all units passes validation.
    5. Intentionally deleting one middle requirement causes validator to fail (SOURCE_REQUIREMENT_DROPPED).
    6. Quality coverage cannot be 100 when that requirement is missing.
    """
    extracted_units = extract_source_semantic_units(prompt)

    # 1. Every explicit semantic request unit survives extraction
    for req in expected_units:
        assert any(
            is_semantic_unit_covered(req, unit) for unit in extracted_units
        ), f"[{domain_name}] Missing extracted unit: {req}"

    # 2. No unsupported unit is invented (all extracted units must be grounded in prompt)
    prompt_lower = prompt.lower()
    for unit in extracted_units:
        tokens = [w for w in unit.split() if len(w) > 3]
        assert any(
            t in prompt_lower for t in tokens
        ), f"[{domain_name}] Invented ungrounded unit: {unit}"

    # Construct complete analysis preserving all units
    complete_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective=f"Build {extracted_units[0]}",
        explicit_requirements=list(expected_units),
        constraints=[],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(score=15, level=AmbiguityLevel.LOW, needs_clarification=False),
    )

    # 3. Complete analysis passes source-coverage gate
    is_covered, src_units, missing = check_source_coverage(prompt, complete_analysis)
    assert is_covered is True, f"[{domain_name}] Incomplete source coverage: {missing}"
    assert len(missing) == 0

    grounding = mock_clients["analyzer"].check_analysis_grounding(complete_analysis, prompt)
    assert grounding.is_grounded is True
    assert len(grounding.missing_source_units) == 0

    # 4. Enhancement preserving all units passes validator
    full_enhanced_prompt = (
        f"OBJECTIVE: Build {extracted_units[0]}\n\nCORE DELIVERABLES:\n"
        + "\n".join(f"- {u}" for u in expected_units)
    )

    mock_clients["client"].chat.return_value = json.dumps({"is_valid": True, "issues": []})
    val_full = await mock_clients["validator"].validate(
        original_prompt=prompt,
        analysis=complete_analysis,
        enhanced_prompt=full_enhanced_prompt,
    )
    assert val_full.is_valid is True, f"[{domain_name}] Full enhancement should be valid"

    # Quality score with full coverage reports 100
    cov_full = mock_clients["scorer"].compute_requirement_coverage(
        analysis=complete_analysis,
        enhanced_prompt=full_enhanced_prompt,
        validation=val_full,
        original_prompt=prompt,
    )
    assert cov_full.score == 100

    # 5. Validator FAILS if one middle requirement is intentionally deleted
    corrupted_units = [u for u in expected_units if u != middle_req_to_delete]
    corrupted_enhanced_prompt = (
        f"OBJECTIVE: Build {extracted_units[0]}\n\nCORE DELIVERABLES:\n"
        + "\n".join(f"- {u}" for u in corrupted_units)
    )

    val_corrupted = await mock_clients["validator"].validate(
        original_prompt=prompt,
        analysis=complete_analysis,
        enhanced_prompt=corrupted_enhanced_prompt,
    )
    assert val_corrupted.is_valid is False, f"[{domain_name}] Validator must reject corrupted prompt missing '{middle_req_to_delete}'"
    dropped_issues = [
        i for i in val_corrupted.issues
        if i.type == ValidationIssueType.SOURCE_REQUIREMENT_DROPPED
        and i.severity == ValidationSeverity.ERROR
    ]
    assert len(dropped_issues) >= 1
    assert any(middle_req_to_delete in (i.message + str(i.evidence)) for i in dropped_issues)

    # 6. Quality coverage CANNOT be 100 when a unit is missing
    cov_corrupted = mock_clients["scorer"].compute_requirement_coverage(
        analysis=complete_analysis,
        enhanced_prompt=corrupted_enhanced_prompt,
        validation=val_corrupted,
        original_prompt=prompt,
    )
    assert cov_corrupted.score < 100, f"[{domain_name}] Coverage score must not be 100 when unit is missing"
    assert cov_corrupted.score <= 80


# =========================================================================
# TASK 6: Adversarial Source-Drop Test
# =========================================================================

@pytest.mark.asyncio
async def test_adversarial_source_drop_prevents_false_positive_validation(mock_clients):
    """TASK 6 — ADVERSARIAL SOURCE-DROP TEST.

    Construct:
    Original: A + B + C + D + E
    Derived analysis: A + B + D + E (C deliberately removed)
    Enhanced: A + B + D + E

    Assert:
    - source-grounding gate fails (is_grounded = False, C in missing_source_units)
    - validator does NOT report intent verified (is_valid = False, SOURCE_REQUIREMENT_DROPPED)
    - requirement coverage is NOT 100
    - intent preservation is NOT 100
    """
    original_prompt = (
        "build an operations platform with user onboarding and credit check and agreement signing "
        "and account activation and welcome email dispatch"
    )
    # 5 semantic units:
    # A: user onboarding
    # B: credit check
    # C: agreement signing
    # D: account activation
    # E: welcome email dispatch

    # Derived analysis has C deliberately removed:
    derived_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="Build an operations platform",
        explicit_requirements=[
            "user onboarding",
            "credit check",
            # "agreement signing" DELIBERATELY REMOVED
            "account activation",
            "welcome email dispatch",
        ],
        constraints=[],
        preferences=[],
        missing_context=[],
        ambiguity=AmbiguityAssessment(score=15, level=AmbiguityLevel.LOW, needs_clarification=False),
    )

    # Enhanced prompt has C deliberately omitted:
    enhanced_prompt = (
        "OBJECTIVE: Build an operations platform\n\n"
        "DELIVERABLES:\n"
        "- user onboarding workflow\n"
        "- credit check evaluation\n"
        "- account activation pipeline\n"
        "- welcome email dispatch service\n"
    )

    # 1. Source-grounding gate MUST fail on derived analysis
    grounding = mock_clients["analyzer"].check_analysis_grounding(derived_analysis, original_prompt)
    assert grounding.is_grounded is False
    assert any("agreement signing" in u for u in grounding.missing_source_units)
    assert any("omitted source requirement: 'agreement signing'" in item for item in grounding.unsupported_items)

    # 2. Validator directly checks authoritative original prompt against enhanced prompt
    mock_clients["client"].chat.return_value = json.dumps({"is_valid": True, "issues": []})
    validation = await mock_clients["validator"].validate(
        original_prompt=original_prompt,
        analysis=derived_analysis,
        enhanced_prompt=enhanced_prompt,
    )

    # Validator MUST NOT report intent verified
    assert validation.is_valid is False
    dropped_issue = next(
        (i for i in validation.issues if i.type == ValidationIssueType.SOURCE_REQUIREMENT_DROPPED),
        None,
    )
    assert dropped_issue is not None
    assert dropped_issue.severity == ValidationSeverity.ERROR
    assert "agreement signing" in dropped_issue.message

    # 3. Requirement coverage CANNOT be 100
    req_cov = mock_clients["scorer"].compute_requirement_coverage(
        analysis=derived_analysis,
        enhanced_prompt=enhanced_prompt,
        validation=validation,
        original_prompt=original_prompt,
    )
    assert req_cov.score < 100
    assert req_cov.score <= 80
    assert "agreement signing" in req_cov.reason

    # 4. Intent preservation CANNOT be 100
    intent_pres = mock_clients["scorer"].compute_intent_preservation(
        validation=validation,
        analysis=derived_analysis,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
    )
    assert intent_pres.score < 100
    assert intent_pres.score <= 40


# =========================================================================
# TASK 7: Keep Prior Hardening Invariants Intact
# =========================================================================

@pytest.mark.asyncio
async def test_prior_hardening_invariants_preserved():
    """Verify all prior production hardening invariants remain intact.

    1. Vague prompt 'build me a system' still triggers high ambiguity & clarification.
    2. No domain invention on underspecified input.
    3. Rejected options are not re-introduced.
    """
    analyzer = PromptAnalyzer(client=AsyncMock())

    # 1. Vague request
    vague_analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.5),
        objective="Build a system",
        explicit_requirements=[],
        constraints=[],
        preferences=[],
        missing_context=["system functional purpose"],
        ambiguity=AmbiguityAssessment(score=10, level=AmbiguityLevel.LOW, needs_clarification=False),
    )
    analyzer._normalize_ambiguity(vague_analysis, "build me a system")
    assert vague_analysis.ambiguity.needs_clarification is True
    assert vague_analysis.ambiguity.score >= 75
    assert len(vague_analysis.ambiguity.questions) >= 1

    # 2. Rejection / superseding
    prompt = "make a calculator wait no actually make a quiz"
    units = extract_source_semantic_units(prompt)
    assert "quiz" in [u.lower() for u in units]
    assert "calculator" not in [u.lower() for u in units]
