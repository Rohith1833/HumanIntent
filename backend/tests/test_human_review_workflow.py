"""Tests for the Auditable Human Review Workflow in Phase 11.

Covers:
1. Review queue export and provenance preservation.
2. Provisional AI proposal recording (strictly PENDING, unverified, reviewer=None).
3. Human auditor verification (APPROVE, REVISE, REJECT, accept-proposal) with required reviewer identity.
4. Validation of revised targets against original user prompts.
5. Strict blocking of unverified provisional / pending records in final export.
6. Zero seed-family leakage in train/validation split.
7. Separation of clarification pairs from standard SFT pairs.
8. Audit report and manifest generation.
"""

import json
import os
import sys
import tempfile
import pytest

# Ensure backend and root are available in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dataset_factory.review import (
    accept_proposal,
    apply_provisional_proposal,
    apply_review_decision,
    batch_apply_proposals,
    batch_apply_reviews,
    export_human_reviewed_dataset,
    export_review_queue,
    generate_review_report,
    load_review_queue,
    split_reviewed_records,
    validate_review_target,
)
from dataset_factory.schemas import HumanReviewDecision, HumanReviewItem


@pytest.fixture
def sample_candidates_file(tmp_path):
    """Fixture providing a mock candidates.jsonl file with multiple seeds/categories."""
    cands = [
        {
            "record_id": "cand_001",
            "category": "CODING",
            "original_prompt": "Write a python function to calculate fibonacci numbers.",
            "enhanced_prompt": "Provide a clean, efficient Python function to calculate Fibonacci numbers with type annotations and docstrings.",
            "curation_status": "ACCEPTED",
            "clarification_required": False,
            "clarification_questions": [],
            "provenance": {"seed_id": "coding_001"},
            "quality": {"quality_score": 0.95},
        },
        {
            "record_id": "cand_002",
            "category": "CODING",
            "original_prompt": "Write a python function to calculate prime numbers.",
            "enhanced_prompt": "Provide a clean, efficient Python function to calculate prime numbers up to n.",
            "curation_status": "ACCEPTED",
            "clarification_required": False,
            "clarification_questions": [],
            "provenance": {"seed_id": "coding_001"},  # Same seed family
            "quality": {"quality_score": 0.92},
        },
        {
            "record_id": "cand_003",
            "category": "CODING",
            "original_prompt": "Write a rust CLI tool for log parsing.",
            "enhanced_prompt": "Develop a high-performance Rust command line tool to parse access logs.",
            "curation_status": "ACCEPTED",
            "clarification_required": False,
            "clarification_questions": [],
            "provenance": {"seed_id": "coding_002"},  # Different seed family
            "quality": {"quality_score": 0.88},
        },
        {
            "record_id": "cand_004",
            "category": "SUMMARIZATION",
            "original_prompt": "Summarize the quarterly financial report highlighting key revenue drivers.",
            "enhanced_prompt": "Analyze and summarize the quarterly financial report, focusing on revenue drivers and operating margins.",
            "curation_status": "ACCEPTED",
            "clarification_required": False,
            "clarification_questions": [],
            "provenance": {"seed_id": "summ_001"},
            "quality": {"quality_score": 0.85},
        },
        {
            "record_id": "cand_005",
            "category": "SUMMARIZATION",
            "original_prompt": "Summarize this vague request without any text.",
            "enhanced_prompt": "Please clarify what document you want summarized.",
            "curation_status": "ACCEPTED",
            "clarification_required": True,
            "clarification_questions": ["What is the source text to summarize?"],
            "provenance": {"seed_id": "summ_002"},
            "quality": {"quality_score": 0.75},
        },
        {
            "record_id": "cand_006",
            "category": "DATA_ANALYSIS",
            "original_prompt": "Calculate monthly churn rate from the attached sales table.",
            "enhanced_prompt": "Calculate monthly churn rate using the sales table provided in the prompt.",
            "curation_status": "ACCEPTED",
            "clarification_required": False,
            "clarification_questions": [],
            "provenance": {"seed_id": "data_001"},
            "quality": {"quality_score": 0.70},
        },
    ]

    fpath = tmp_path / "candidates.jsonl"
    with open(fpath, "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    return str(fpath)


def test_export_review_queue_and_decision_lifecycle(sample_candidates_file, tmp_path):
    """Test exporting queue, applying decisions, and preserving reviewer history."""
    queue_path = str(tmp_path / "review_queue.jsonl")

    # 1. Export queue
    count = export_review_queue(
        source_path_or_dir=sample_candidates_file,
        output_queue_path=queue_path,
        dataset_version="test-v1",
    )
    assert count == 6

    items = load_review_queue(queue_path)
    assert len(items) == 6
    for item in items:
        assert item.decision == HumanReviewDecision.PENDING
        assert item.verified_by_human is False
        assert item.reviewer is None

    # 2. Approve cand_001 by real human auditor
    item1 = apply_review_decision(
        queue_path=queue_path,
        record_id="cand_001",
        decision="APPROVE",
        reviewer="auditor_alice",
        reason="Clean and training-grade unchanged",
    )
    assert item1.decision == HumanReviewDecision.APPROVE
    assert item1.verified_by_human is True
    assert item1.final_target == item1.teacher_target
    assert item1.reviewer == "auditor_alice"

    # 3. Reject cand_006 due to missing source data
    item6 = apply_review_decision(
        queue_path=queue_path,
        record_id="cand_006",
        decision="REJECT",
        reviewer="auditor_bob",
        reason="SOURCE_MISSING: Prompt references sales table but none provided",
    )
    assert item6.decision == HumanReviewDecision.REJECT
    assert item6.verified_by_human is True
    assert item6.reason.startswith("SOURCE_MISSING")

    # 4. Reject without reason must raise ValueError
    with pytest.raises(ValueError, match="rejection reason is required"):
        apply_review_decision(
            queue_path=queue_path,
            record_id="cand_004",
            decision="REJECT",
            reviewer="auditor_alice",
            reason="",
        )

    # 5. Review without human reviewer must raise ValueError
    with pytest.raises(ValueError, match="valid human reviewer identity is required"):
        apply_review_decision(
            queue_path=queue_path,
            record_id="cand_004",
            decision="APPROVE",
            reviewer=None,
        )


def test_provisional_ai_proposals_distinguished_from_human_verification(sample_candidates_file, tmp_path):
    """Test that provisional AI proposals are strictly distinguished from verified human decisions."""
    queue_path = str(tmp_path / "review_queue.jsonl")
    export_review_queue(sample_candidates_file, queue_path, "test-v1")

    valid_revised = (
        "Write an idiomatic, well-documented Rust command line utility to parse access logs with structured output."
    )

    # 1. Record an AI-generated proposal
    prop_item = apply_provisional_proposal(
        queue_path=queue_path,
        record_id="cand_003",
        proposal_decision="REVISE",
        proposed_target=valid_revised,
        proposal_reason="AI defect analysis: Cleaned up repetitive preamble",
        proposal_source="ai_provisional_reviewer",
    )

    # CRITICAL INVARIANT: The record must remain strictly PENDING human verification
    assert prop_item.decision == HumanReviewDecision.PENDING
    assert prop_item.verified_by_human is False
    assert prop_item.reviewer is None
    assert prop_item.revised_target is None  # Not yet admitted!
    assert prop_item.final_target is None    # Unadmitted returns None

    # Proposal content is tracked separately
    assert prop_item.provisional_decision == HumanReviewDecision.REVISE
    assert prop_item.proposed_target == valid_revised
    assert prop_item.proposal_source == "ai_provisional_reviewer"

    # 2. Strict export must reject provisional proposals without human verification
    out_dir = str(tmp_path / "unverified_export")
    with pytest.raises(ValueError, match="records are still PENDING human verification"):
        export_human_reviewed_dataset(
            queue_path=queue_path,
            output_dir=out_dir,
            dataset_version="test-v1",
            strict=True,
        )

    # 3. Real human auditor reviews and accepts the proposal
    accepted_item = accept_proposal(
        queue_path=queue_path,
        record_id="cand_003",
        reviewer="dr_elizabeth_chen",
        reason="Verified technical accuracy of Rust CLI correction",
    )

    # Now certified by a human
    assert accepted_item.decision == HumanReviewDecision.REVISE
    assert accepted_item.verified_by_human is True
    assert accepted_item.reviewer == "dr_elizabeth_chen"
    assert accepted_item.revised_target == valid_revised
    assert accepted_item.final_target == valid_revised


def test_revise_target_validation(sample_candidates_file, tmp_path):
    """Test that revised targets are validated against user prompt before admission."""
    queue_path = str(tmp_path / "review_queue.jsonl")
    export_review_queue(sample_candidates_file, queue_path, "test-v1")

    # A valid revision
    valid_revised = (
        "Write an idiomatic, well-documented Rust command line utility to parse access logs with structured output."
    )
    rev_item = apply_review_decision(
        queue_path=queue_path,
        record_id="cand_003",
        decision="REVISE",
        revised_target=valid_revised,
        reason="Cleaned up repetitive preamble and clarified structured output format",
        reviewer="auditor_alice",
    )
    assert rev_item.decision == HumanReviewDecision.REVISE
    assert rev_item.final_target == valid_revised
    assert rev_item.teacher_target != valid_revised  # Original preserved

    # An invalid revision that contradicts user intent (e.g., changes Rust CLI log parsing into Python game)
    invalid_revised = "Write a Python Pygame script for a 2D space shooter game."
    with pytest.raises(ValueError, match="Revised target failed curation validation"):
        apply_review_decision(
            queue_path=queue_path,
            record_id="cand_003",
            decision="REVISE",
            revised_target=invalid_revised,
            reason="Contradictory revision",
            reviewer="auditor_alice",
        )


def test_strict_governance_blocks_pending_records(sample_candidates_file, tmp_path):
    """Test that unreviewed (PENDING) records are strictly blocked from dataset export."""
    queue_path = str(tmp_path / "review_queue.jsonl")
    export_review_queue(sample_candidates_file, queue_path, "test-v1")

    # Only 1 reviewed, 5 pending
    apply_review_decision(
        queue_path=queue_path,
        record_id="cand_001",
        decision="APPROVE",
        reviewer="auditor_alice",
    )

    out_dir = str(tmp_path / "export_out")
    with pytest.raises(ValueError, match="records are still PENDING human verification"):
        export_human_reviewed_dataset(
            queue_path=queue_path,
            output_dir=out_dir,
            dataset_version="test-v1",
            strict=True,
        )


def test_seed_grouped_zero_leakage_split(tmp_path):
    """Test that train/val splitting never leaks records from the same seed family across splits."""
    items = [
        # Coding seed 1 (2 records)
        HumanReviewItem(
            record_id="c1",
            dataset_version="v1",
            category="CODING",
            seed_id="coding_001",
            original_prompt="prompt 1",
            teacher_target="target 1",
            decision=HumanReviewDecision.APPROVE,
            verified_by_human=True,
            reviewer="auditor_alice",
        ),
        HumanReviewItem(
            record_id="c2",
            dataset_version="v1",
            category="CODING",
            seed_id="coding_001",
            original_prompt="prompt 2",
            teacher_target="target 2",
            decision=HumanReviewDecision.APPROVE,
            verified_by_human=True,
            reviewer="auditor_alice",
        ),
        # Coding seed 2 (1 record)
        HumanReviewItem(
            record_id="c3",
            dataset_version="v1",
            category="CODING",
            seed_id="coding_002",
            original_prompt="prompt 3",
            teacher_target="target 3",
            decision=HumanReviewDecision.APPROVE,
            verified_by_human=True,
            reviewer="auditor_alice",
        ),
        # Coding seed 3 (1 record)
        HumanReviewItem(
            record_id="c4",
            dataset_version="v1",
            category="CODING",
            seed_id="coding_003",
            original_prompt="prompt 4",
            teacher_target="target 4",
            decision=HumanReviewDecision.APPROVE,
            verified_by_human=True,
            reviewer="auditor_alice",
        ),
        # Planning seed 1 (1 record)
        HumanReviewItem(
            record_id="p1",
            dataset_version="v1",
            category="PLANNING",
            seed_id="planning_001",
            original_prompt="plan 1",
            teacher_target="target p1",
            decision=HumanReviewDecision.APPROVE,
            verified_by_human=True,
            reviewer="auditor_alice",
        ),
        # Planning seed 2 (1 record)
        HumanReviewItem(
            record_id="p2",
            dataset_version="v1",
            category="PLANNING",
            seed_id="planning_002",
            original_prompt="plan 2",
            teacher_target="target p2",
            decision=HumanReviewDecision.APPROVE,
            verified_by_human=True,
            reviewer="auditor_alice",
        ),
    ]

    train, val = split_reviewed_records(items, train_ratio=0.7, random_seed=42)

    assert len(train) > 0
    assert len(val) > 0
    assert len(train) + len(val) == len(items)

    train_seeds = {r.seed_id for r in train}
    val_seeds = {r.seed_id for r in val}

    # CRITICAL INVARIANT: Zero seed-family intersection!
    seed_intersection = train_seeds.intersection(val_seeds)
    assert len(seed_intersection) == 0, f"Seed family leaked between train and val: {seed_intersection}"


def test_full_export_and_clarification_separation(sample_candidates_file, tmp_path):
    """Test exporting dataset with clarification separation, rejected records, manifest, and card."""
    queue_path = str(tmp_path / "review_queue.jsonl")
    export_review_queue(sample_candidates_file, queue_path, "test-v1")

    # Review all 6 records with explicit human auditor identity
    decisions = [
        {"record_id": "cand_001", "decision": "APPROVE", "reason": "Approved unchanged", "reviewer": "auditor_lead"},
        {"record_id": "cand_002", "decision": "APPROVE", "reason": "Approved unchanged", "reviewer": "auditor_lead"},
        {
            "record_id": "cand_003",
            "decision": "REVISE",
            "revised_target": "Write an idiomatic, well-documented Rust command line utility to parse access logs with structured output.",
            "reason": "Cleaned up prompt preamble",
            "reviewer": "auditor_lead",
        },
        {"record_id": "cand_004", "decision": "APPROVE", "reason": "Approved unchanged", "reviewer": "auditor_lead"},
        {"record_id": "cand_005", "decision": "REJECT", "reason": "TASK_UNSUPPORTED: Vague summarization prompt", "reviewer": "auditor_lead"},
        {"record_id": "cand_006", "decision": "REJECT", "reason": "SOURCE_MISSING: Missing sales table", "reviewer": "auditor_lead"},
    ]
    batch_res = batch_apply_reviews(queue_path, decisions, reviewer="auditor_lead")
    assert batch_res["applied_count"] == 6

    out_dir = str(tmp_path / "final_export")
    res = export_human_reviewed_dataset(
        queue_path=queue_path,
        output_dir=out_dir,
        dataset_version="humanintent-v0.2.4-test",
        strict=True,
    )

    assert res["status"] == "SUCCESS"
    assert os.path.exists(os.path.join(out_dir, "sft_train.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "sft_validation.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "rejected_records.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "review_audit.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "manifest.json"))
    assert os.path.exists(os.path.join(out_dir, "DATASET_CARD.md"))

    # Check rejected records
    with open(os.path.join(out_dir, "rejected_records.jsonl"), "r", encoding="utf-8") as f:
        rejected_lines = [json.loads(line) for line in f if line.strip()]
    assert len(rejected_lines) == 2
    rej_ids = {r["record_id"] for r in rejected_lines}
    assert rej_ids == {"cand_005", "cand_006"}

    # Check manifest statistics
    manifest = res["manifest"]
    assert manifest["approved_count"] == 3
    assert manifest["revised_count"] == 1
    assert manifest["rejected_count"] == 2
    assert manifest["pending_count"] == 0
    assert manifest["split_counts"]["rejected"] == 2
    assert manifest["split_counts"]["sft_train"] + manifest["split_counts"]["sft_validation"] == 4
    assert manifest["reviewers"] == ["auditor_lead"]
