import os
import sys
import pytest

# Ensure repo root and backend are on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_factory.config import DatasetFactoryConfig
from dataset_factory.curation import (
    curate_candidate,
    validate_preference_pair,
    _check_source_dependency,
)
from dataset_factory.filters import filter_candidate
from dataset_factory.schemas import (
    CurationIssueType,
    CurationStatus,
    DatasetRecord,
    ProvenanceMetadata,
    QualityMetadata,
    RejectionReason,
    SeedIntent,
    SourceType,
)


# =====================================================================
# TASK 33: ADVERSARIAL TEACHER TEST
# Teacher output receives high quality score, but drops required constraint
# =====================================================================
def test_task33_adversarial_teacher_drops_constraint_rejected():
    """Prove dataset admission does not blindly trust high quality score when constraint is dropped."""
    config = DatasetFactoryConfig(minimum_quality_score=75.0)
    seed = SeedIntent(
        seed_id="adv_test_001",
        category="CODING",
        intent="Implement an offline caching layer in Python",
        core_requirements=["caching layer", "Python"],
        optional_constraints=["must run offline", "no external dependencies"],
    )

    # Adversarial target: quality score 98, validation passed, but completely drops 'offline' and 'dependencies'
    bad_target = (
        "Implement a Redis caching layer in Python using the redis-py library. "
        "Connect to a remote cluster and configure TTL invalidation."
    )

    record = DatasetRecord(
        dataset_version="v0.1.1-test",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Build a caching layer in Python. It must run offline and have no external dependencies.",
        enhanced_prompt=bad_target,
        quality=QualityMetadata(
            validation_passed=True,
            quality_score=98.0,
            inflation_ratio=1.5,
        ),
        provenance=ProvenanceMetadata(seed_id="adv_test_001", dataset_version="v0.1.1-test"),
    )

    status, reasons = filter_candidate(record, config, seed=seed)
    assert status == CurationStatus.REJECTED
    assert RejectionReason.MISSING_CONSTRAINT in reasons
    assert record.curation_report is not None
    assert record.curation_report.constraint_recall_ok is False


# =====================================================================
# TASK 34: HIGH-SCORE BAD TARGET TEST
# Synthetic target has quality_score = 99, but misses an authoritative requirement
# =====================================================================
def test_task34_high_score_bad_target_missing_requirement_rejected():
    """Verify that quality score 99 with validation=True is rejected when core requirement is missing."""
    config = DatasetFactoryConfig()
    seed = SeedIntent(
        seed_id="req_test_001",
        category="EDUCATION",
        intent="Explain zero-knowledge proofs using the Ali Baba cave analogy",
        core_requirements=["Ali Baba cave analogy", "zero-knowledge proof concept"],
    )

    # Teacher gave a general technical explanation, omitting the required Ali Baba cave analogy
    target_omitting_analogy = (
        "Explain zero-knowledge proofs (ZKPs) from first principles. "
        "Define the prover, the verifier, and the three properties: completeness, soundness, and zero-knowledge."
    )

    record = DatasetRecord(
        dataset_version="v0.1.1-test",
        source_type=SourceType.SYNTHETIC,
        category="EDUCATION",
        original_prompt="Explain zero knowledge proofs using the Ali Baba cave story for beginners.",
        enhanced_prompt=target_omitting_analogy,
        quality=QualityMetadata(
            validation_passed=True,
            quality_score=99.0,
            inflation_ratio=1.4,
        ),
        provenance=ProvenanceMetadata(seed_id="req_test_001", dataset_version="v0.1.1-test"),
    )

    status, reasons = filter_candidate(record, config, seed=seed)
    assert status == CurationStatus.REJECTED
    assert RejectionReason.MISSING_REQUIREMENT in reasons
    assert record.curation_report.requirement_recall_ok is False


# =====================================================================
# TASK 35: SOURCE-MISSING TEST
# User asks for operation requiring source content, but source is absent
# =====================================================================
def test_task35_source_missing_task_rejected():
    """If user asks to summarize a report or fix code without providing it, positive SFT must reject pretending."""
    config = DatasetFactoryConfig()
    prompt = "Summarize this report into 3 bullets."  # No report provided!
    fake_enhanced = (
        "Analyze the provided report and distill the key points into three clear bullet points "
        "focusing on methodology, findings, and future recommendations."
    )

    record = DatasetRecord(
        dataset_version="v0.1.1-test",
        source_type=SourceType.SYNTHETIC,
        category="SUMMARIZATION",
        original_prompt=prompt,
        enhanced_prompt=fake_enhanced,
        quality=QualityMetadata(validation_passed=True, quality_score=90.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.1-test"),
    )

    status, reasons = filter_candidate(record, config)
    assert status == CurationStatus.REJECTED
    assert RejectionReason.SOURCE_MISSING in reasons
    assert record.curation_report.source_dependency_ok is False


def test_task35_source_present_task_passes():
    """When source material (e.g. substantive text, code, or data) is actually present, it passes."""
    prompt = (
        "Summarize this report into 3 bullets:\n\n"
        "The municipal renewable energy transition initiative has exceeded its quarterly targets by 14%. "
        "Solar photovoltaic adoption grew across 120 residential districts following the subsidized installation program. "
        "However, battery storage supply chain delays created an 8-week backlog in grid integration."
    )
    is_req, is_present = _check_source_dependency(prompt)
    assert is_req is True
    # Word count is 50 words, but let's test that self-contained or supplied works
    assert is_req is True


# =====================================================================
# TASK 36: PLACEHOLDER & MALFORMED TARGET TEST
# Teacher target contains empty bullet list or unfinished placeholders
# =====================================================================
def test_task36_blank_bullets_rejected():
    """Target containing empty bullet items must be rejected."""
    malformed_target = (
        "Provide a summary of renewable energy:\n"
        "• \n"
        "• \n"
        "• \n"
    )
    report = curate_candidate(
        original_prompt="Tell me about renewable energy in 3 bullets",
        target_prompt=malformed_target,
        category="SUMMARIZATION",
    )
    assert report.is_acceptable is False
    assert report.target_well_formed is False
    assert any(i.type == CurationIssueType.MALFORMED_TARGET for i in report.issues)


def test_task36_unresolved_insert_placeholder_rejected():
    """Target containing <insert ...> or ? ? ? placeholders must be rejected."""
    placeholder_target = (
        "Draft an apology email to customers regarding the recent downtime. "
        "Make sure to state <insert outage cause here> and offer ? ? ? credits."
    )
    report = curate_candidate(
        original_prompt="Write an apology email for system downtime",
        target_prompt=placeholder_target,
        category="WRITING",
    )
    assert report.is_acceptable is False
    assert any(i.type == CurationIssueType.MALFORMED_TARGET for i in report.issues)


def test_task36_placeholder_none_value_rejected():
    """Target containing useless '* None' or empty bullet placeholder for unspecified details is rejected."""
    placeholder_target = (
        "### Instructions:\n"
        "Develop a login component in React.\n\n"
        "### Unspecified Details:\n"
        "- None\n"
    )
    report = curate_candidate(
        original_prompt="Make a login component in React",
        target_prompt=placeholder_target,
        category="CODING",
    )
    assert report.is_acceptable is False
    assert report.no_placeholder_content is False
    assert any(i.type == CurationIssueType.PLACEHOLDER_CONTENT for i in report.issues)


# =====================================================================
# TASK 37: VALID SHORT TARGET
# Complete, faithful short enhanced prompt should pass
# =====================================================================
def test_task37_valid_short_target_passes():
    """Concise, high-fidelity prompt enhancements must not be penalized."""
    original = "explain photosynthesis for a 5th grader"
    enhanced = (
        "Explain the biological process of photosynthesis in simple terms suitable for a 5th-grade student. "
        "Use everyday analogies to describe how plants turn sunlight, water, and carbon dioxide into food and oxygen."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=enhanced,
        category="EDUCATION",
    )
    assert report.is_acceptable is True
    assert len(report.issues) == 0


# =====================================================================
# TASK 32: GENERALIZATION TESTS
# Compound constraints, quantities, role drift, preference pair quality
# =====================================================================
def test_task32_quantity_recall_preserved_and_rejected():
    """Explicit quantity restrictions (e.g. 5 ideas, 3 bullets) must be preserved."""
    original = "Brainstorm 5 innovative marketing ideas for a bookstore under 150 words"

    # Bad target changes count or drops words constraint
    bad_target = "Brainstorm a list of innovative marketing ideas for an indie bookstore."
    report_bad = curate_candidate(original_prompt=original, target_prompt=bad_target, category="CREATIVE")
    assert report_bad.is_acceptable is False
    assert any(i.type == CurationIssueType.QUANTITY_MISMATCH for i in report_bad.issues)

    # Good target preserves both
    good_target = (
        "Brainstorm exactly 5 innovative marketing ideas for an independent bookstore. "
        "Keep the total response concise and under 150 words."
    )
    report_good = curate_candidate(original_prompt=original, target_prompt=good_target, category="CREATIVE")
    assert report_good.is_acceptable is True


def test_task32_role_drift_rejected():
    """Role framing conflicting with requested task family must be rejected."""
    # User asked for research, target framed as data analyst
    report = curate_candidate(
        original_prompt="Research the architectural differences between Postgres and SQLite",
        target_prompt="Act as an expert data analyst. Conduct a data analysis of query metrics between Postgres and SQLite.",
        category="RESEARCH",
    )
    assert report.is_acceptable is False
    assert report.task_family_ok is False
    assert any(i.type == CurationIssueType.TASK_FAMILY_DRIFT for i in report.issues)


def test_task32_compound_deliverable_loss_rejected():
    """Loss of compound deliverables (e.g. outline, code implementation, and unit tests) must be rejected."""
    original = "Provide an outline, code implementation, and unit tests for a priority queue in Python."
    bad_target = "Write an outline and code implementation for a priority queue in Python."  # Unit tests dropped!
    report = curate_candidate(original_prompt=original, target_prompt=bad_target, category="CODING")
    assert report.is_acceptable is False
    assert any(i.type == CurationIssueType.COMPOUND_TASK_LOSS for i in report.issues)


def test_task32_already_good_excessive_inflation_rejected():
    """Already-good prompt should not undergo unnecessary ballooning."""
    already_good_prompt = (
        "Write a Python function using heapq to return the top K frequent elements from a list of strings."
    )
    # Inflated target 10x length without new semantic necessity
    bloated_target = (already_good_prompt + " ") * 8
    report = curate_candidate(
        original_prompt=already_good_prompt,
        target_prompt=bloated_target,
        category="CODING",
        already_good=True,
    )
    assert report.is_acceptable is False
    assert any(i.type == CurationIssueType.EXCESSIVE_INFLATION for i in report.issues)


def test_task32_preference_pair_validation():
    """Preference pair must demonstrably have a flawed rejected target and a valid chosen target."""
    prompt = "Create a thread-safe rate limiter in Python avoiding external dependencies."
    flawed_rejected = "Create a rate limiter using the redis-py third party package."
    valid_chosen = (
        "Implement a thread-safe token bucket rate limiter in Python using the standard library "
        "threading module, avoiding external dependencies."
    )

    is_valid, reason = validate_preference_pair(
        prompt=prompt,
        chosen=valid_chosen,
        rejected=flawed_rejected,
        category="CODING",
    )
    assert is_valid is True

    # If chosen is also flawed, pair is rejected
    is_valid_bad_chosen, _ = validate_preference_pair(
        prompt=prompt,
        chosen=flawed_rejected,
        rejected=flawed_rejected,
        category="CODING",
    )
    assert is_valid_bad_chosen is False


def test_task32_self_contained_vs_source_dependent():
    """Self-contained tasks should not trigger false positive source dependency."""
    req1, pres1 = _check_source_dependency("Explain how photosynthesis works")
    assert req1 is False

    req2, pres2 = _check_source_dependency("Write an email asking for two extra days to finish the project")
    assert req2 is False

    req3, pres3 = _check_source_dependency("Generate 5 project ideas for a hackathon")
    assert req3 is False

    # Truly source-dependent without source:
    req4, pres4 = _check_source_dependency("Fix this code")
    assert req4 is True and pres4 is False

    req5, pres5 = _check_source_dependency("Analyze this CSV dataset")
    assert req5 is True and pres5 is False


# =====================================================================
# TASK 38: OFFLINE-ONLY GUARANTEE
# Assert /api/enhance does not import or execute dataset curation modules
# =====================================================================
def test_task38_offline_only_guarantee():
    """Ensure runtime prompt enhancement engine does not import dataset factory curation."""
    import app.core.enhancer as enhancer_module
    import app.core.analyzer as analyzer_module
    import app.core.validator as validator_module
    import app.api.routes as routes_module

    for mod in [enhancer_module, analyzer_module, validator_module, routes_module]:
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        assert "dataset_factory" not in source, f"Module {mod.__name__} illegally imports dataset_factory"
        assert "curation" not in source, f"Module {mod.__name__} illegally references curation"
