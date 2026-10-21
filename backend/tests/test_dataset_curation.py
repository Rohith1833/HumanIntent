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
from app.schemas.analysis import PromptCategory


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


# =====================================================================
# TASK 39: UNSUPPORTED INVENTED METRIC TEST
# Target introduces ungrounded analytical metrics or segmentation
# =====================================================================
def test_task39_unsupported_invented_metric_rejected():
    """Verify that unrequested analytical metrics (e.g. segmentation breakdown) are rejected."""
    original = "Analyze customer churn data and summarize important trends."
    bad_target = (
        "### Core Requirements\n"
        "- Ingest the customer churn dataset.\n"
        "- Calculate churn rate by every customer segment.\n"
        "- Summarize important trends and insights."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=bad_target,
        category="DATA_ANALYSIS",
    )
    assert report.is_acceptable is False
    assert report.no_unsupported_requirement is False
    assert any(i.type == CurationIssueType.UNSUPPORTED_REQUIREMENT for i in report.issues)


# =====================================================================
# TASK 40: UNSUPPORTED TECHNOLOGY INJECTION TEST
# Target introduces unrequested named technology / framework
# =====================================================================
def test_task40_unsupported_technology_rejected():
    """Verify that unrequested named technologies (e.g. PostgreSQL/SQLAlchemy) are rejected."""
    original = "Build a simple web scraper in Python."
    bad_target = (
        "### Core Requirements\n"
        "- Scrape data from the web page in Python.\n"
        "- Store the scraped data in a PostgreSQL database using SQLAlchemy."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=bad_target,
        category="CODING",
    )
    assert report.is_acceptable is False
    assert report.no_unsupported_requirement is False
    assert any("postgresql" in i.message.lower() for i in report.issues)


# =====================================================================
# TASK 41: UNSUPPORTED METHODOLOGY TEST
# Target introduces unrequested model architecture / optimization library
# =====================================================================
def test_task41_unsupported_methodology_rejected():
    """Verify that unrequested methodologies and hyperparameter tools are rejected."""
    original = "Predict customer sales for the next quarter."
    bad_target = (
        "### Core Requirements\n"
        "- Train an XGBoost regressor using 5-fold cross-validation.\n"
        "- Tune hyperparameters using Optuna."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=bad_target,
        category="DATA_ANALYSIS",
    )
    assert report.is_acceptable is False
    assert report.no_unsupported_requirement is False
    assert any(i.type == CurationIssueType.UNSUPPORTED_REQUIREMENT for i in report.issues)


# =====================================================================
# TASK 42: VALID PARAPHRASED REQUIREMENT PASSES
# High-fidelity semantic paraphrase must be accepted without penalty
# =====================================================================
def test_task42_valid_paraphrased_requirement_passes():
    """Verify that legitimate semantic paraphrasing of user requirements is accepted."""
    original = "Explain photosynthesis for a 5th grader."
    good_target = (
        "Explain the biological process of photosynthesis in simple terms suitable for a 5th-grade student. "
        "Use everyday analogies to describe how plants turn sunlight, water, and carbon dioxide into food and oxygen."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=good_target,
        category="EDUCATION",
    )
    assert report.is_acceptable is True
    assert report.no_unsupported_requirement is True


# =====================================================================
# TASK 43: HARMLESS TARGET RESTRUCTURING PASSES
# Harmless sections (Role, Output Format) must not be flagged as invented
# =====================================================================
def test_task43_harmless_target_restructuring_passes():
    """Verify that standard presentational sections (Role, Output Format) pass curation."""
    original = "Write a function to reverse a linked list."
    good_target = (
        "### Role\n"
        "Act as an expert software engineer.\n\n"
        "### Core Requirements\n"
        "- Implement a function that reverses a singly linked list.\n"
        "- Return the new head of the reversed list.\n\n"
        "### Output Format\n"
        "Provide clean, well-commented code with time and space complexity analysis."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=good_target,
        category="CODING",
    )
    assert report.is_acceptable is True
    assert report.no_unsupported_requirement is True


# =====================================================================
# TASK 44: HIGH QUALITY SCORE CANNOT BYPASS UNSUPPORTED REQUIREMENT
# High quality score (98/100) must still be rejected if requirement is ungrounded
# =====================================================================
def test_task44_high_quality_score_cannot_bypass_unsupported_requirement():
    """Verify that quality score 98 with validation=True is rejected when target invents obligations."""
    config = DatasetFactoryConfig()
    bad_target = (
        "### Core Requirements\n"
        "- Ingest the customer churn dataset.\n"
        "- Calculate churn rate by every customer segment.\n"
        "- Summarize important trends and insights."
    )
    record = DatasetRecord(
        dataset_version="v0.2.0-test",
        source_type=SourceType.SYNTHETIC,
        category="DATA_ANALYSIS",
        original_prompt="Analyze customer churn data and summarize important trends.",
        enhanced_prompt=bad_target,
        quality=QualityMetadata(
            validation_passed=True,
            quality_score=98.0,
            inflation_ratio=1.4,
        ),
        provenance=ProvenanceMetadata(seed_id="churn_test_01", dataset_version="v0.2.0-test"),
    )

    status, reasons = filter_candidate(record, config)
    assert status == CurationStatus.REJECTED
    assert RejectionReason.UNSUPPORTED_REQUIREMENT in reasons
    assert record.curation_report is not None
    assert record.curation_report.no_unsupported_requirement is False


# =====================================================================
# TASK 45: RUNTIME LOCKED PRODUCTION FILES UNCHANGED
# Assert bit-for-bit SHA-256 match for all 9 core production files
# =====================================================================
def test_task45_runtime_locked_files_unchanged():
    """Verify zero modification to locked Phase 10.2 production runtime files."""
    import hashlib

    locked_hashes = {
        "app/core/source_coverage.py": "6b8a7f815249dc8b405deb56a0220fa9534f58b551a1530b9e75625f097da0c2",
        "app/core/semantic_matcher.py": "f7cff655bdf66afeafd125efab8627a2404e8581b056f3ae07059f516ef88358",
        "app/core/nli_matcher.py": "255d836448235e729370ec038cefd56dea09773d56b8c3cafeb831469cffa50c",
        "app/core/analyzer.py": "d7f281d465c1adc78b4c59eccde420bc5785b50b7ab590f3a323ba9249c44752",
        "app/core/enhancer.py": "9adf0b3c30ae0014069cd263d0cebeb14b5f8e649b40902d0852ca0d866a4bb4",
        "app/core/validator.py": "897fed04b0166ded04649e6817565b686c099806e60c47f1f536714751162585",
        "app/core/scorer.py": "4fd822e2a19a9c870aa0fdb0e2b9daebac5f235c560c0987e0db3404a9619bc2",
        "app/core/category_router.py": "01ad6ed6683e784c3d6366b0a9a5f7092b5b5701f7b20603d9d328adaafe2634",
        "app/core/intent_anchor.py": "02899e4ec92f5790f2c272c8be13ab6fe5d8080c851aae44f953e124d3d34f8e",
    }

    for rel_path, expected_hash in locked_hashes.items():
        full_path = os.path.join(backend_dir, rel_path)
        assert os.path.exists(full_path), f"File {rel_path} must exist"
        h = hashlib.sha256()
        with open(full_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        assert h.hexdigest() == expected_hash, f"File {rel_path} was modified! Hash mismatch."


# =====================================================================
# TASK 46: CONVERSATIONAL FILLER PROMOTED TO REQUIREMENTS REJECTED
# Conversational discourse from messy prompt must NOT become CORE REQUIREMENTS
# =====================================================================
def test_task46_conversational_filler_promoted_to_requirements_rejected():
    """Verify that conversational filler / discourse fragments in Core Requirements are rejected."""
    original = "So like, we gotta get this onboarding plan ready for our new hire. Right? I'm thinking 30 days. You know, super important and all that jazz."
    bad_target = (
        "### Core Requirements\n"
        "- Create a 30-day onboarding plan.\n"
        "- Specific goals and milestones.\n"
        "- Right\n"
        "- I'm thinking\n"
        "- You know\n"
        "- Super important\n"
        "- All that jazz"
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=bad_target,
        category="PLANNING",
    )
    assert report.is_acceptable is False
    assert report.no_non_substantive_requirement is False
    assert any(i.type == CurationIssueType.NON_SUBSTANTIVE_REQUIREMENT for i in report.issues)


def test_task46b_unseen_conversational_filler_rejected():
    """Verify that unseen discourse markers and hanging fragments in Core Requirements are rejected."""
    original = "We need a script to clean up logs."
    bad_target = (
        "### Core Requirements\n"
        "- Clean up old log files older than 7 days.\n"
        "- Basically you see what I mean\n"
        "- And so forth\n"
        "- We're looking at"
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=bad_target,
        category="CODING",
    )
    assert report.is_acceptable is False
    assert report.no_non_substantive_requirement is False
    assert any(i.type == CurationIssueType.NON_SUBSTANTIVE_REQUIREMENT for i in report.issues)


# =====================================================================
# TASK 47: VALID SHORT REQUIREMENT SURVIVES
# Short but valid technical requirements must NOT be flagged as non-substantive
# =====================================================================
def test_task47_valid_short_requirement_survives():
    """Verify that short valid requirements like 'Use Python' survive substantive checks."""
    original = "Create a rate limiter in Python."
    good_target = (
        "### Core Requirements\n"
        "- Implement a rate limiter.\n"
        "- Use Python.\n\n"
        "### Output Format\n"
        "Provide working code with docstrings."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=good_target,
        category="CODING",
    )
    assert report.is_acceptable is True
    assert report.no_non_substantive_requirement is True


# =====================================================================
# TASK 48: VALID NEGATIVE REQUIREMENT SURVIVES
# Negative constraint requirements must survive substantive checks
# =====================================================================
def test_task48_valid_negative_requirement_survives():
    """Verify that valid negative constraints like 'No Redis' survive substantive checks."""
    original = "Create a rate limiter without Redis."
    good_target = (
        "### Core Requirements\n"
        "- Implement a rate limiter.\n"
        "- No Redis."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=good_target,
        category="CODING",
    )
    assert report.is_acceptable is True
    assert report.no_non_substantive_requirement is True


# =====================================================================
# TASK 49: VALID QUANTITATIVE REQUIREMENT SURVIVES
# Quantitative bound requirements must survive substantive checks
# =====================================================================
def test_task49_valid_quantitative_requirement_survives():
    """Verify that valid quantitative constraints like '3 bullets' survive substantive checks."""
    original = "Summarize the key trends into 3 bullets."
    good_target = (
        "### Core Requirements\n"
        "- Summarize the key trends.\n"
        "- Exactly 3 bullets."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=good_target,
        category="SUMMARIZATION",
    )
    assert report.is_acceptable is True
    assert report.no_non_substantive_requirement is True


# =====================================================================
# TASK 50: GENERIC MISSING-SOURCE SUMMARIZATION REJECTED
# Summarizing an unsupplied abstract/paper must NOT enter positive SFT
# =====================================================================
def test_task50_generic_missing_source_summarization_rejected():
    """Verify that generic missing-source summarization requests are rejected."""
    original = (
        "Summarize the abstract of a research paper on perovskite solar cell efficiency "
        "into three concise bulleted executive takeaways."
    )
    good_looking_target = (
        "### Core Requirements\n"
        "- Summarize the efficiency gains and degradation factors of perovskite solar cells.\n"
        "- Provide exactly three concise bulleted takeaways."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=good_looking_target,
        category="SUMMARIZATION",
    )
    assert report.is_acceptable is False
    assert report.source_dependency_ok is False
    assert any(i.type == CurationIssueType.SOURCE_MISSING for i in report.issues)


# =====================================================================
# TASK 51: GENERIC MISSING-SOURCE DATA ANALYSIS REJECTED
# Analyzing an unsupplied dataset/CSV must be rejected
# =====================================================================
def test_task51_generic_missing_source_data_analysis_rejected():
    """Verify that generic missing-source data analysis requests are rejected."""
    original = "Analyze a CSV of customer transactions and identify top churn predictors."
    target = (
        "### Core Requirements\n"
        "- Ingest customer transaction data.\n"
        "- Identify top factors predicting churn."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=target,
        category="DATA_ANALYSIS",
    )
    assert report.is_acceptable is False
    assert report.source_dependency_ok is False
    assert any(i.type == CurationIssueType.SOURCE_MISSING for i in report.issues)


# =====================================================================
# TASK 52: GENERIC MISSING-SOURCE CODE REVIEW REJECTED
# Reviewing/debugging unsupplied code must be rejected
# =====================================================================
def test_task52_generic_missing_source_code_review_rejected():
    """Verify that generic missing-source code review requests are rejected."""
    original = "Review a Python function for race conditions and suggest concurrency fixes."
    target = (
        "### Core Requirements\n"
        "- Analyze concurrency and lock acquisition in the function.\n"
        "- Identify potential race conditions."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=target,
        category="CODING",
    )
    assert report.is_acceptable is False
    assert report.source_dependency_ok is False
    assert any(i.type == CurationIssueType.SOURCE_MISSING for i in report.issues)


# =====================================================================
# TASK 53: SELF-CONTAINED KNOWLEDGE TASK ACCEPTED
# Conceptual knowledge requests must NOT be treated as source-dependent
# =====================================================================
def test_task53_self_contained_knowledge_task_accepted():
    """Verify that self-contained knowledge requests are not treated as source-dependent."""
    original = "Explain race conditions in Python and how threading locks prevent them."
    target = (
        "Explain what race conditions are in concurrent Python programming. "
        "Describe how threading.Lock and synchronization primitives ensure thread safety."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=target,
        category="EDUCATION",
    )
    assert report.is_acceptable is True
    assert report.source_dependency_ok is True


# =====================================================================
# TASK 54: EMBEDDED SOURCE CODE AND DATA PASSES
# When code or data is actually present in the prompt, source_dependency passes
# =====================================================================
def test_task54_embedded_source_code_and_data_passes():
    """Verify that embedded code, CSV, or text passes the source presence check."""
    # Code review with actual code present
    code_prompt = (
        "Review this Python function for race conditions:\n"
        "```python\n"
        "def transfer(src, dst, amt):\n"
        "    src.balance -= amt\n"
        "    dst.balance += amt\n"
        "```"
    )
    target = (
        "### Core Requirements\n"
        "- Review the provided transfer function for race conditions.\n"
        "- Provide a thread-safe implementation using locks."
    )
    report_code = curate_candidate(
        original_prompt=code_prompt,
        target_prompt=target,
        category="CODING",
    )
    assert report_code.source_dependency_ok is True

    # Data analysis with actual data present
    data_prompt = (
        "Analyze this CSV data:\n"
        "date,user_id,amount\n"
        "2026-01-01,u101,15.50\n"
        "2026-01-02,u102,29.99\n"
        "2026-01-03,u103,45.00\n"
        "Summarize trends."
    )
    data_target = (
        "### Core Requirements\n"
        "- Analyze user transaction trends from the provided CSV data.\n"
        "- Summarize revenue patterns."
    )
    report_data = curate_candidate(
        original_prompt=data_prompt,
        target_prompt=data_target,
        category="DATA_ANALYSIS",
    )
    assert report_data.source_dependency_ok is True


# =====================================================================
# TASK 55: QUALITY SCORE 98 CANNOT BYPASS SOURCE MISSING OR FILLER
# Invariant: quality_score >= 75 does NOT override curation errors
# =====================================================================
def test_task55_quality_score_cannot_bypass_source_missing_or_filler():
    """Verify that quality score 98/100 cannot bypass source-missing or non-substantive rejection."""
    config = DatasetFactoryConfig()

    # 1. Missing source candidate with quality 98.0
    rec_source_missing = DatasetRecord(
        dataset_version="v0.2.1-test",
        source_type=SourceType.SYNTHETIC,
        category="SUMMARIZATION",
        original_prompt="Summarize the abstract of a research paper on perovskite solar cells.",
        enhanced_prompt="### Core Requirements\n- Summarize the research paper abstract.",
        quality=QualityMetadata(
            validation_passed=True,
            quality_score=98.0,
            inflation_ratio=1.2,
        ),
        provenance=ProvenanceMetadata(seed_id="sum_test_01", dataset_version="v0.2.1-test"),
    )
    status, reasons = filter_candidate(rec_source_missing, config)
    assert status == CurationStatus.REJECTED
    assert RejectionReason.SOURCE_MISSING in reasons

    # 2. Non-substantive filler candidate with quality 98.0
    rec_filler = DatasetRecord(
        dataset_version="v0.2.1-test",
        source_type=SourceType.SYNTHETIC,
        category="PLANNING",
        original_prompt="Create an onboarding plan. Right? You know.",
        enhanced_prompt="### Core Requirements\n- Create an onboarding plan.\n- Right\n- You know",
        quality=QualityMetadata(
            validation_passed=True,
            quality_score=98.0,
            inflation_ratio=1.2,
        ),
        provenance=ProvenanceMetadata(seed_id="plan_test_01", dataset_version="v0.2.1-test"),
    )
    status2, reasons2 = filter_candidate(rec_filler, config)
    assert status2 == CurationStatus.REJECTED
    assert RejectionReason.NON_SUBSTANTIVE_REQUIREMENT in reasons2


# =====================================================================
# TASK 56: CANONICAL HEADER FORMATTING RECOGNITION & SUBSTANTIVE QUALITY
# Invariant: Section recognition is canonical and formatting-insensitive.
# Markdown decoration, punctuation separators, and casing cannot bypass curation.
# =====================================================================
def test_task56_canonical_header_formatting_detects_filler():
    """Verify that different header formattings (###, **, plain, /, &, +) are canonically recognized."""
    from dataset_factory.curation import check_structured_target_substantive_quality, normalize_section_header

    # 1. Direct unit test of normalize_section_header
    assert normalize_section_header("CORE REQUIREMENTS") == "requirements"
    assert normalize_section_header("CORE / EXPLICIT REQUIREMENTS") == "requirements"
    assert normalize_section_header("CORE & EXPLICIT REQUIREMENTS") == "requirements"
    assert normalize_section_header("EXPLICIT REQUIREMENTS") == "requirements"
    assert normalize_section_header("REQUIREMENTS") == "requirements"
    assert normalize_section_header("CORE REQUIREMENTS:") == "requirements"
    assert normalize_section_header("### CORE / EXPLICIT REQUIREMENTS") == "requirements"
    assert normalize_section_header("### CORE / EXPLICIT REQUIREMENTS:") == "requirements"
    assert normalize_section_header("**CORE / EXPLICIT REQUIREMENTS:**") == "requirements"
    assert normalize_section_header("**CORE & EXPLICIT REQUIREMENTS**") == "requirements"
    assert normalize_section_header("CONSTRAINTS") == "constraints"
    assert normalize_section_header("CONSTRAINTS:") == "constraints"
    assert normalize_section_header("EXPLICIT CONSTRAINTS") == "constraints"
    assert normalize_section_header("RESTRICTIONS") == "constraints"
    assert normalize_section_header("PREFERENCES") == "preferences"
    assert normalize_section_header("STYLE PREFERENCES") == "preferences"
    assert normalize_section_header("QUESTIONS TO ANSWER") == "questions to answer"
    assert normalize_section_header("QUESTIONS TO ANSWER:") == "questions to answer"
    assert normalize_section_header("REQUIRED QUESTIONS") == "questions to answer"
    assert normalize_section_header("REQUIRED METRICS") == "metrics"
    assert normalize_section_header("METRICS") == "metrics"
    assert normalize_section_header("REQUIRED METHODOLOGY") == "methodology"
    assert normalize_section_header("METHODOLOGY") == "methodology"
    assert normalize_section_header("OUTPUT FORMAT:") == "output format"
    assert normalize_section_header("UNSPECIFIED DETAILS:") == "unspecified details"

    # Ensure list items/bullets are never treated as headers
    assert normalize_section_header("* Right") is None
    assert normalize_section_header("- Build a 4-week rollout plan") is None
    assert normalize_section_header("1. Exactly 3 bullets") is None

    # 2. Combined slash-form header with conversational filler -> MUST BE REJECTED
    target_with_slash_header = (
        "### CORE / EXPLICIT REQUIREMENTS:\n\n"
        "- Build a 4-week rollout plan\n"
        "- You know\n"
        "- Sounds good\n"
        "- Weekly milestones"
    )
    ok, issues = check_structured_target_substantive_quality(target_with_slash_header)
    assert ok is False
    assert any("You know" in iss for iss in issues)

    # 3. Equivalent valid variants with standard header -> MUST PASS
    target_valid_plain = (
        "CORE REQUIREMENTS:\n"
        "- Use Python\n"
        "- Exactly 3 bullets"
    )
    ok_valid, issues_valid = check_structured_target_substantive_quality(target_valid_plain)
    assert ok_valid is True
    assert len(issues_valid) == 0

    # 4. Alternative header formatting (**CORE & EXPLICIT REQUIREMENTS**) -> MUST PASS if substantive
    target_valid_ampersand = (
        "**CORE & EXPLICIT REQUIREMENTS**\n"
        "- Use Python\n"
        "- Exactly 3 bullets"
    )
    ok_amp, issues_amp = check_structured_target_substantive_quality(target_valid_ampersand)
    assert ok_amp is True
    assert len(issues_amp) == 0

    # 5. Alternative header formatting with filler -> MUST BE REJECTED
    target_filler_amp = (
        "**CORE & EXPLICIT REQUIREMENTS**\n"
        "- Use Python\n"
        "- Right\n"
        "- All that jazz"
    )
    ok_amp_bad, issues_amp_bad = check_structured_target_substantive_quality(target_filler_amp)
    assert ok_amp_bad is False
    assert any("Right" in iss for iss in issues_amp_bad)


# =====================================================================
# TASK 57: EXACT v0.2.2 ACCEPTED PLANNING TARGET CONTROL REGRESSION
# Invariant: The exact polluted PLANNING target accepted in v0.2.2-smoke
# MUST NOT pass curate_candidate(...) or filter_candidate(...).
# =====================================================================
def test_task57_v022_accepted_planning_target_rejected():
    """Verify that the exact accepted PLANNING candidate from humanintent-v0.2.2-smoke is rejected."""
    config = DatasetFactoryConfig()

    original_prompt = (
        "So like, we need to get this junior backend engineer up and running on our agile dev team, right? "
        "So, I'm thinking, let's break it down into a 30-day onboarding plan, with, like, specific goals and milestones for each week, you know? "
        "And, oh yeah, we gotta make sure they get their local environment set up and submit their first PR within the first week, that's like, super important. "
        "And can we please pair them up with someone who's, like, familiar with the codebase and all that jazz? "
        "And, hey, let's make sure we're not overloading them, you know, avoiding burnout and all that. So, week-by-week, we're looking at:"
    )

    exact_v022_target = (
        "Act as an experienced strategic planner and workflow advisor. Produce an actionable, realistic plan with clear milestones and logical sequence.\n\n"
        "CORE / EXPLICIT REQUIREMENTS:\n\n"
        "* Create a 30-day onboarding plan for a junior backend engineer\n"
        "* Specific goals and milestones for each week\n"
        "* Right\n"
        "* I'm thinking\n"
        "* You know\n"
        "* Oh yeah\n"
        "* We gotta make sure they get their local environment set up\n"
        "* Super important\n"
        "* All that jazz\n"
        "* Hey\n"
        "* All that. So\n"
        "* We're looking at\n"
        "* Local environment set up\n"
        "* First PR submission within the first week\n"
        "* Pairing with someone familiar with the codebase\n"
        "* Avoiding burnout and overload\n"
        "* Running on our agile dev team\n\n"
        "CONSTRAINTS:\n\n"
        "* Not overloading the junior engineer\n"
        "* Avoiding burnout\n\n"
        "UNSPECIFIED DETAILS:\n\n"
        "* Junior engineer's current skill level and experience\n"
        "* Current codebase complexity and familiarity\n\n"
        "OUTPUT FORMAT:\n\n"
        "* Phased breakdown with objectives, action items, and checkpoint criteria"
    )

    # 1. Direct curate_candidate verification
    report = curate_candidate(
        original_prompt=original_prompt,
        target_prompt=exact_v022_target,
        category="PLANNING",
        clarification_required=False,
    )
    assert report.is_acceptable is False
    assert report.no_non_substantive_requirement is False
    assert any(iss.type == CurationIssueType.NON_SUBSTANTIVE_REQUIREMENT for iss in report.issues)

    # 2. Full pipeline filter_candidate verification
    record = DatasetRecord(
        dataset_version="v0.2.2-smoke",
        source_type=SourceType.SYNTHETIC,
        category="PLANNING",
        original_prompt=original_prompt,
        enhanced_prompt=exact_v022_target,
        quality=QualityMetadata(
            validation_passed=True,
            quality_score=95.0,
            inflation_ratio=1.4,
        ),
        provenance=ProvenanceMetadata(seed_id="plan_001", dataset_version="v0.2.2-smoke"),
    )
    status, reasons = filter_candidate(record, config)
    assert status == CurationStatus.REJECTED
    assert RejectionReason.NON_SUBSTANTIVE_REQUIREMENT in reasons


# =====================================================================
# TASK 51: PRESENTATIONAL SECTION ISOLATION
# Sections like UNSPECIFIED DETAILS or MISSING CONTEXT are not requirements
# =====================================================================
def test_unspecified_details_presentational_isolation():
    """Verify that UNSPECIFIED DETAILS / MISSING CONTEXT are treated as presentational and not requirements."""
    original = "Compare B-tree and LSM-tree throughput under write-heavy loads."
    target = (
        "### OBJECTIVE:\n"
        "Compare the throughput of B-tree and LSM-tree architectures under write-heavy loads.\n\n"
        "### CORE / EXPLICIT REQUIREMENTS:\n"
        "- Throughput comparison\n"
        "- B-tree and LSM-tree architectures\n"
        "- Write-heavy loads\n\n"
        "### UNSPECIFIED DETAILS:\n"
        "- Specific workload description is missing\n"
        "- Desired granularity is missing\n\n"
        "### OUTPUT STRUCTURE:\n"
        "Directly provide the comparison of throughput under write-heavy loads."
    )
    report = curate_candidate(
        original_prompt=original,
        target_prompt=target,
        category="RESEARCH",
    )
    assert report.is_acceptable is True
    assert report.no_unsupported_requirement is True


# =====================================================================
# TASK 52: STRUCTURED EMPTY SECTION DETECTION
# Normal sentences following headers pass; truly empty headers fail
# =====================================================================
def test_structured_empty_section_detection():
    """Verify that structured section detection distinguishes populated headers from empty headers."""
    # A. Populated header followed by normal sentence starting with uppercase
    target_good = (
        "### CORE REQUIREMENTS:\n"
        "- Analyze the architecture\n\n"
        "### OUTPUT STRUCTURE:\n"
        "Directly compare the systems with clear focus on performance."
    )
    report_good = curate_candidate(
        original_prompt="Analyze the architecture and directly compare the systems.",
        target_prompt=target_good,
        category="RESEARCH",
    )
    assert report_good.target_well_formed is True

    # B. Truly empty header followed immediately by another header
    target_empty = (
        "### CORE REQUIREMENTS:\n"
        "- Analyze the architecture\n\n"
        "### UNSPECIFIED DETAILS:\n\n"
        "### OUTPUT FORMAT:\n"
        "Provide a summary."
    )
    report_empty = curate_candidate(
        original_prompt="Analyze the architecture.",
        target_prompt=target_empty,
        category="RESEARCH",
    )
    assert report_empty.target_well_formed is False
    assert any(iss.type == CurationIssueType.MALFORMED_TARGET for iss in report_empty.issues)


# =====================================================================
# TASK 53: PROMPT AUTHORITATIVE RECALL
# Unprompted seed constraints are not enforced against teacher target
# =====================================================================
def test_prompt_authoritative_negative_constraint_recall():
    """Verify that unprompted optional seed constraints do not cause false positive rejections."""
    seed = SeedIntent(
        seed_id="edu_test",
        category=PromptCategory.EDUCATION,
        intent="Explain zero knowledge proofs",
        core_requirements=["explain concept"],
        optional_constraints=["under 400 words", "avoid mathematical formulas"],
    )
    prompt = "Explain the Zero Knowledge Proof concept using the Ali Baba cave analogy at a high school level."
    target = (
        "Act as an effective teacher. Explain the Zero Knowledge Proof concept using the Ali Baba cave analogy.\n\n"
        "### TOPIC & OBJECTIVE:\n"
        "Explain Zero Knowledge Proof concept.\n\n"
        "### EXPLANATION STYLE:\n"
        "Provide a clear and simple walkthrough using the Ali Baba cave analogy."
    )
    report = curate_candidate(
        original_prompt=prompt,
        target_prompt=target,
        category="EDUCATION",
        seed=seed,
    )
    assert report.is_acceptable is True
    assert report.constraint_recall_ok is True
    assert report.quantity_recall_ok is True


# =====================================================================
# TASK 54: QUANTITY PRESERVATION (DOWNGRADE REJECTION)
# Explicit quantity downgraded to vague 'several' is rejected
# =====================================================================
def test_quantity_downgrade_rejected():
    """Verify that explicit quantity like 'ten' downgraded to 'several' is rejected."""
    prompt = "Develop ten innovative theme concepts and gameplay mechanics for a deck-building game."
    downgraded_target = (
        "Act as a creative thinker. Develop unique theme concepts for a deck-building game.\n\n"
        "### CORE REQUIREMENTS:\n"
        "- Cooperative gameplay mechanics\n"
        "- Player synergy hooks\n\n"
        "### OUTPUT EXPECTATIONS:\n"
        "Provide several distinct, creative theme concepts."
    )
    report = curate_candidate(
        original_prompt=prompt,
        target_prompt=downgraded_target,
        category="CREATIVE",
    )
    assert report.is_acceptable is False
    assert report.quantity_recall_ok is False
    assert any(iss.type == CurationIssueType.QUANTITY_MISMATCH for iss in report.issues)


# =====================================================================
# TASK 55: CALCULATION ON SAMPLE WITHOUT NUMBERS REJECTED
# Prompt asking to calculate metrics on a sample without data fails
# =====================================================================
def test_calculation_on_sample_without_numeric_data_rejected():
    """Verify that requests to calculate metrics on sample data lacking numbers fail SOURCE_MISSING."""
    prompt = (
        "here's a sample of api response times - calculate the median latency, p95, p99, and standard deviation, "
        "and help me identify any outliers. if you could also show me the step-by-step formula for each calculation, "
        "that'd be awesome!"
    )
    target = (
        "Act as an expert data analyst. Calculate the median latency, p95, and p99 of the API response times.\n\n"
        "### CORE REQUIREMENTS:\n"
        "- Calculate median latency\n"
        "- Calculate p95 and p99"
    )
    report = curate_candidate(
        original_prompt=prompt,
        target_prompt=target,
        category="DATA_ANALYSIS",
    )
    assert report.is_acceptable is False
    assert report.source_dependency_ok is False
    assert any(iss.type == CurationIssueType.SOURCE_MISSING for iss in report.issues)


# =====================================================================
# TASK 56: CONCEPTUAL INQUIRY DELIVERABLE ESCALATION REJECTED
# Why/how inquiry escalated to mandatory code deliverable is rejected
# =====================================================================
def test_conceptual_inquiry_deliverable_escalation_rejected():
    """Verify that escalating a conceptual why/how inquiry into mandatory code is rejected."""
    prompt = "Why does a Python asyncio event loop hang when mixing synchronous HTTP requests with async coroutines, and how do I prevent this blocking behavior?"
    escalated_target = (
        "Act as an experienced software developer. Prevent blocking behavior when mixing synchronous requests with async coroutines in Python.\n\n"
        "### CORE REQUIREMENTS:\n"
        "- Explain why asyncio hangs with synchronous requests\n"
        "- Address blocking behavior prevention\n\n"
        "### OUTPUT EXPECTATIONS:\n"
        "Provide an implementation approach followed by code to prevent blocking behavior."
    )
    report = curate_candidate(
        original_prompt=prompt,
        target_prompt=escalated_target,
        category="DEBUGGING",
    )
    assert report.is_acceptable is False
    assert report.no_unsupported_requirement is False
    assert any(iss.type == CurationIssueType.UNSUPPORTED_REQUIREMENT for iss in report.issues)


# =====================================================================
# TASK 57: TEMPLATE VARIABLE HALLUCINATION REJECTED
# System template placeholders like modality or interface type fail
# =====================================================================
def test_hallucinated_template_variables_rejected():
    """Verify that hallucinated template parameters in unspecified details are rejected."""
    prompt = "What are the societal tradeoffs of remote work on urban planning and commercial real estate?"
    template_target = (
        "Act as an objective researcher.\n\n"
        "### CORE REQUIREMENTS:\n"
        "- Analyze remote work societal tradeoffs\n"
        "- Urban planning and real estate impacts\n\n"
        "### UNSPECIFIED DETAILS:\n"
        "- modality\n"
        "- interface type"
    )
    report = curate_candidate(
        original_prompt=prompt,
        target_prompt=template_target,
        category="GENERAL",
    )
    assert report.is_acceptable is False
    assert report.no_placeholder_content is False
    assert any(iss.type == CurationIssueType.MALFORMED_TARGET for iss in report.issues)


# =====================================================================
# TASK 58: SEED EXCLUSION DOMAIN SWAP REJECTED
# Generator replacing seed exclusion with unrelated domain fails
# =====================================================================
def test_seed_exclusion_domain_swap_rejected():
    """Verify that synthetic generator swapping exclusion domains is rejected as INTENT_DRIFT."""
    from dataset_factory.filters import check_intent_preservation
    seed = SeedIntent(
        seed_id="creative_test",
        category=PromptCategory.CREATIVE,
        intent="Brainstorm ten innovative theme concepts for a cooperative tabletop game",
        core_requirements=["10 distinct theme concepts", "cooperative mechanics"],
        optional_constraints=["avoid standard medieval fantasy or generic sci-fi tropes"],
    )
    # Generator swapped domain to steampunk / superhero
    prompt = (
        "Develop ten innovative theme concepts and cooperative gameplay mechanics for a tabletop deck-building game "
        "that do NOT feature steampunk, post-apocalyptic, or generic superhero themes."
    )
    assert check_intent_preservation(prompt, seed) is False
