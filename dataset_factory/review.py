import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from dataset_factory.curation import curate_candidate
from dataset_factory.schemas import (
    ClarificationExample,
    CurationStatus,
    DatasetRecord,
    HumanReviewDecision,
    HumanReviewItem,
    HumanReviewManifest,
    SFTExample,
    SFTMessage,
)

logger = logging.getLogger("dataset_factory.review")


def export_review_queue(
    source_path_or_dir: str,
    output_queue_path: str,
    dataset_version: str,
    only_accepted: bool = True,
) -> int:
    """Export candidate input/target pairs to an auditable human review queue.

    Preserves existing reviewer decisions if the output file already exists.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_queue_path)), exist_ok=True)

    # 1. Load any existing reviews to avoid overwriting ongoing work
    existing_reviews: Dict[str, HumanReviewItem] = {}
    if os.path.exists(output_queue_path):
        with open(output_queue_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    item = HumanReviewItem(**json.loads(line_str))
                    existing_reviews[item.record_id] = item

    # 2. Ingest candidates from file or export directory
    records_to_queue: List[Dict[str, Any]] = []

    if os.path.isdir(source_path_or_dir):
        # Read from an existing export directory (sft_train, sft_validation, etc.)
        for fname in ["sft_train.jsonl", "sft_validation.jsonl", "clarification_train.jsonl", "clarification_validation.jsonl"]:
            fpath = os.path.join(source_path_or_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    data = json.loads(line_str)
                    messages = data.get("messages", [])
                    meta = data.get("metadata", {})
                    user_content = messages[0]["content"] if len(messages) > 0 else ""
                    asst_content = messages[1]["content"] if len(messages) > 1 else ""
                    records_to_queue.append({
                        "record_id": meta.get("record_id") or data.get("record_id"),
                        "dataset_version": dataset_version,
                        "category": meta.get("category", "GENERAL"),
                        "seed_id": meta.get("provenance", {}).get("seed_id"),
                        "original_prompt": user_content,
                        "teacher_target": asst_content,
                        "clarification_required": "clarification" in fname,
                        "clarification_questions": [],
                        "quality_score": meta.get("quality_score"),
                        "metadata": meta,
                    })
    elif os.path.isfile(source_path_or_dir):
        # Read from intermediate candidates jsonl
        with open(source_path_or_dir, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                status = data.get("curation_status")
                if only_accepted and status != "ACCEPTED":
                    continue
                records_to_queue.append({
                    "record_id": data.get("record_id"),
                    "dataset_version": dataset_version,
                    "category": data.get("category", "GENERAL"),
                    "seed_id": data.get("provenance", {}).get("seed_id"),
                    "original_prompt": data.get("original_prompt", ""),
                    "teacher_target": data.get("enhanced_prompt"),
                    "clarification_required": data.get("clarification_required", False),
                    "clarification_questions": data.get("clarification_questions", []),
                    "quality_score": data.get("quality", {}).get("quality_score"),
                    "metadata": {
                        "source_type": data.get("source_type"),
                        "provenance": data.get("provenance", {}),
                        "labels": data.get("labels", {}),
                    },
                })
    else:
        raise FileNotFoundError(f"Source path not found: {source_path_or_dir}")

    # 3. Create or preserve review items
    queued_items: List[HumanReviewItem] = []
    seen_ids: Set[str] = set()

    for r in records_to_queue:
        rid = r["record_id"]
        if rid in seen_ids:
            continue
        seen_ids.add(rid)

        if rid in existing_reviews:
            queued_items.append(existing_reviews[rid])
        else:
            queued_items.append(
                HumanReviewItem(
                    record_id=rid,
                    dataset_version=dataset_version,
                    category=r["category"],
                    seed_id=r["seed_id"],
                    original_prompt=r["original_prompt"],
                    teacher_target=r["teacher_target"],
                    clarification_required=r["clarification_required"],
                    clarification_questions=r["clarification_questions"],
                    quality_score=r["quality_score"],
                    decision=HumanReviewDecision.PENDING,
                    metadata=r.get("metadata", {}),
                )
            )

    # Write queue
    with open(output_queue_path, "w", encoding="utf-8") as f:
        for item in queued_items:
            f.write(item.model_dump_json() + "\n")

    logger.info(f"Exported {len(queued_items)} review items to {output_queue_path}")
    return len(queued_items)


def load_review_queue(queue_path: str) -> List[HumanReviewItem]:
    """Load all items from a review queue file."""
    if not os.path.exists(queue_path):
        raise FileNotFoundError(f"Review queue file does not exist: {queue_path}")

    items = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                items.append(HumanReviewItem(**json.loads(line_str)))
    return items


def save_review_queue(queue_path: str, items: List[HumanReviewItem]) -> None:
    """Save review items to a queue file atomically."""
    temp_path = queue_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    os.replace(temp_path, queue_path)


def validate_review_target(
    original_prompt: str,
    target_prompt: str,
    category: str,
) -> Tuple[bool, List[str]]:
    """Validate a revised or approved target prompt against original user intent.

    Ensures human corrections meet all training-data invariants.
    """
    issues: List[str] = []
    if not target_prompt or not target_prompt.strip():
        return False, ["Target prompt cannot be empty."]

    curation_report = curate_candidate(
        original_prompt=original_prompt,
        target_prompt=target_prompt,
        category=category,
        clarification_required=False,
    )

    if not curation_report.is_acceptable:
        for iss in curation_report.issues:
            if iss.severity == "ERROR":
                issues.append(f"[{iss.type.value}] {iss.message}")

    return len(issues) == 0, issues


def apply_provisional_proposal(
    queue_path: str,
    record_id: str,
    proposal_decision: str,
    proposed_target: Optional[str] = None,
    proposal_reason: Optional[str] = None,
    proposal_source: str = "ai_provisional_reviewer",
    validate: bool = True,
) -> HumanReviewItem:
    """Record an AI-generated provisional proposal for human review.

    CRITICAL PROVENANCE INVARIANT:
    The record's human decision remains strictly PENDING (verified_by_human=False, reviewer=None).
    The AI suggestion is preserved in proposed_target and proposal_reason.
    """
    decision_upper = proposal_decision.upper().strip()
    if decision_upper not in [d.value for d in HumanReviewDecision]:
        raise ValueError(f"Invalid proposal decision '{proposal_decision}'. Must be APPROVE, REVISE, or REJECT.")

    dec_enum = HumanReviewDecision(decision_upper)
    items = load_review_queue(queue_path)
    target_item: Optional[HumanReviewItem] = None

    for item in items:
        if item.record_id == record_id or item.record_id.startswith(record_id):
            target_item = item
            break

    if target_item is None:
        raise KeyError(f"Record ID '{record_id}' not found in review queue {queue_path}.")

    # Validate target if revising
    if dec_enum == HumanReviewDecision.REVISE:
        if not proposed_target or not proposed_target.strip():
            raise ValueError("Proposed target is required when proposal decision is REVISE.")
        if validate:
            valid, issues = validate_review_target(
                original_prompt=target_item.original_prompt,
                target_prompt=proposed_target.strip(),
                category=target_item.category,
            )
            if not valid:
                raise ValueError(
                    f"Proposed target failed curation validation:\n" + "\n".join(f"- {i}" for i in issues)
                )

    target_item.provisional_decision = dec_enum
    target_item.proposed_target = proposed_target.strip() if proposed_target else None
    target_item.proposal_reason = proposal_reason.strip() if proposal_reason else None
    target_item.proposal_source = proposal_source

    # Maintain strictly unreviewed human state
    target_item.decision = HumanReviewDecision.PENDING
    target_item.verified_by_human = False
    target_item.reviewer = None
    target_item.reviewed_at = None
    target_item.revised_target = None
    target_item.reason = None

    save_review_queue(queue_path, items)
    logger.info(f"Recorded provisional proposal {dec_enum.value} from {proposal_source} for {target_item.record_id}")
    return target_item


def batch_apply_proposals(
    queue_path: str,
    proposals: List[Dict[str, Any]],
    proposal_source: str = "ai_provisional_reviewer",
    validate: bool = True,
) -> Dict[str, Any]:
    """Bulk record provisional AI proposals without marking any records as human-reviewed."""
    items = load_review_queue(queue_path)
    item_map = {item.record_id: item for item in items}
    prefix_map = {item.record_id[:8]: item for item in items}

    applied_count = 0
    errors: List[str] = []

    for p in proposals:
        rid = p.get("record_id")
        dec = p.get("decision", "").upper().strip()
        prop_target = p.get("revised_target") or p.get("proposed_target")
        reason = p.get("reason") or p.get("proposal_reason")

        target_item = item_map.get(rid) or prefix_map.get(rid[:8] if rid else "")
        if not target_item:
            errors.append(f"Record {rid} not found in queue.")
            continue

        try:
            dec_enum = HumanReviewDecision(dec)
            if dec_enum == HumanReviewDecision.REVISE:
                if not prop_target or not prop_target.strip():
                    errors.append(f"Record {rid}: REVISE requires proposed target.")
                    continue
                if validate:
                    valid, issues = validate_review_target(
                        original_prompt=target_item.original_prompt,
                        target_prompt=prop_target.strip(),
                        category=target_item.category,
                    )
                    if not valid:
                        errors.append(f"Record {rid} validation failed: {issues}")
                        continue
                target_item.proposed_target = prop_target.strip()
            else:
                target_item.proposed_target = None

            target_item.provisional_decision = dec_enum
            target_item.proposal_reason = reason
            target_item.proposal_source = proposal_source

            # Strictly PENDING human verification
            target_item.decision = HumanReviewDecision.PENDING
            target_item.verified_by_human = False
            target_item.reviewer = None
            target_item.reviewed_at = None
            target_item.revised_target = None
            target_item.reason = None

            applied_count += 1
        except Exception as e:
            errors.append(f"Record {rid}: {str(e)}")

    save_review_queue(queue_path, items)
    return {
        "applied_count": applied_count,
        "total_specified": len(proposals),
        "errors": errors,
    }


def accept_proposal(
    queue_path: str,
    record_id: str,
    reviewer: str,
    reason: Optional[str] = None,
    validate: bool = True,
) -> HumanReviewItem:
    """Human auditor accepts and certifies an AI provisional proposal."""
    if not reviewer or not reviewer.strip() or reviewer.strip() == "ai_provisional_reviewer":
        raise ValueError("A valid human reviewer identity is required to accept a proposal.")

    items = load_review_queue(queue_path)
    target_item: Optional[HumanReviewItem] = None

    for item in items:
        if item.record_id == record_id or item.record_id.startswith(record_id):
            target_item = item
            break

    if target_item is None:
        raise KeyError(f"Record ID '{record_id}' not found in review queue {queue_path}.")

    if not target_item.provisional_decision:
        raise ValueError(f"Record '{record_id}' has no provisional AI proposal to accept.")

    dec_enum = target_item.provisional_decision
    if dec_enum == HumanReviewDecision.REVISE:
        if not target_item.proposed_target:
            raise ValueError(f"Record '{record_id}' has provisional REVISE decision but missing proposed_target.")
        if validate:
            valid, issues = validate_review_target(
                original_prompt=target_item.original_prompt,
                target_prompt=target_item.proposed_target,
                category=target_item.category,
            )
            if not valid:
                raise ValueError(f"Proposed target failed validation: {issues}")
        target_item.revised_target = target_item.proposed_target
    else:
        target_item.revised_target = None

    target_item.decision = dec_enum
    target_item.reviewer = reviewer.strip()
    target_item.reason = reason.strip() if reason else target_item.proposal_reason
    target_item.reviewed_at = datetime.now(timezone.utc).isoformat()
    target_item.verified_by_human = True

    save_review_queue(queue_path, items)
    logger.info(f"Human auditor '{reviewer}' accepted proposal {dec_enum.value} for record {target_item.record_id}")
    return target_item


def apply_review_decision(
    queue_path: str,
    record_id: str,
    decision: str,
    revised_target: Optional[str] = None,
    reason: Optional[str] = None,
    reviewer: Optional[str] = None,
    validate: bool = True,
) -> HumanReviewItem:
    """Apply an explicit human review decision to a record in the queue.

    Enforces validation of revised targets, required reasons, and real human reviewer provenance.
    """
    if not reviewer or not reviewer.strip() or reviewer.strip() == "ai_provisional_reviewer":
        raise ValueError("A valid human reviewer identity is required for review decisions.")

    decision_upper = decision.upper().strip()
    if decision_upper not in [d.value for d in HumanReviewDecision]:
        raise ValueError(f"Invalid review decision '{decision}'. Must be APPROVE, REVISE, or REJECT.")

    dec_enum = HumanReviewDecision(decision_upper)

    if dec_enum == HumanReviewDecision.REVISE:
        if not revised_target or not revised_target.strip():
            raise ValueError("Revised target is required when decision is REVISE.")
        if not reason or not reason.strip():
            raise ValueError("A reason / revision note is required when decision is REVISE.")
    elif dec_enum == HumanReviewDecision.REJECT:
        if not reason or not reason.strip():
            raise ValueError("A rejection reason is required when decision is REJECT.")

    items = load_review_queue(queue_path)
    target_item: Optional[HumanReviewItem] = None

    for item in items:
        if item.record_id == record_id or item.record_id.startswith(record_id):
            target_item = item
            break

    if target_item is None:
        raise KeyError(f"Record ID '{record_id}' not found in review queue {queue_path}.")

    # Validate target if revising
    if dec_enum == HumanReviewDecision.REVISE and validate:
        valid, issues = validate_review_target(
            original_prompt=target_item.original_prompt,
            target_prompt=revised_target.strip(),
            category=target_item.category,
        )
        if not valid:
            raise ValueError(
                f"Revised target failed curation validation:\n" + "\n".join(f"- {i}" for i in issues)
            )

    # Update item with human verification
    target_item.decision = dec_enum
    target_item.reviewer = reviewer.strip()
    target_item.reason = reason.strip() if reason else ("Verified training-grade" if dec_enum == HumanReviewDecision.APPROVE else "")
    target_item.reviewed_at = datetime.now(timezone.utc).isoformat()
    target_item.verified_by_human = True

    if dec_enum == HumanReviewDecision.REVISE:
        target_item.revised_target = revised_target.strip()
    elif dec_enum == HumanReviewDecision.APPROVE:
        target_item.revised_target = None

    save_review_queue(queue_path, items)
    logger.info(f"Applied human decision {dec_enum.value} to record {target_item.record_id} by {reviewer}")
    return target_item


def batch_apply_reviews(
    queue_path: str,
    decisions: List[Dict[str, Any]],
    reviewer: Optional[str] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Bulk apply verified human decisions from a list of review specifications."""
    items = load_review_queue(queue_path)
    item_map = {item.record_id: item for item in items}
    prefix_map = {item.record_id[:8]: item for item in items}

    applied_count = 0
    errors: List[str] = []

    for d in decisions:
        rid = d.get("record_id")
        dec = d.get("decision", "").upper().strip()
        rev_target = d.get("revised_target")
        reason = d.get("reason")
        rev_name = d.get("reviewer") or reviewer

        if not rev_name or not rev_name.strip() or rev_name.strip() == "ai_provisional_reviewer":
            errors.append(f"Record {rid}: A valid human reviewer identity is required.")
            continue

        target_item = item_map.get(rid) or prefix_map.get(rid[:8] if rid else "")
        if not target_item:
            errors.append(f"Record {rid} not found in queue.")
            continue

        try:
            dec_enum = HumanReviewDecision(dec)
            if dec_enum == HumanReviewDecision.REVISE:
                if not rev_target or not rev_target.strip():
                    errors.append(f"Record {rid}: REVISE requires revised_target.")
                    continue
                if not reason:
                    errors.append(f"Record {rid}: REVISE requires reason.")
                    continue
                if validate:
                    valid, issues = validate_review_target(
                        original_prompt=target_item.original_prompt,
                        target_prompt=rev_target.strip(),
                        category=target_item.category,
                    )
                    if not valid:
                        errors.append(f"Record {rid} validation failed: {issues}")
                        continue
                target_item.revised_target = rev_target.strip()
            elif dec_enum == HumanReviewDecision.REJECT:
                if not reason:
                    errors.append(f"Record {rid}: REJECT requires reason.")
                    continue
                target_item.revised_target = None
            elif dec_enum == HumanReviewDecision.APPROVE:
                target_item.revised_target = None
                if not reason:
                    reason = "Verified training-grade"

            target_item.decision = dec_enum
            target_item.reviewer = rev_name.strip()
            target_item.reason = reason
            target_item.reviewed_at = datetime.now(timezone.utc).isoformat()
            target_item.verified_by_human = True
            applied_count += 1
        except Exception as e:
            errors.append(f"Record {rid}: {str(e)}")

    save_review_queue(queue_path, items)
    return {
        "applied_count": applied_count,
        "total_specified": len(decisions),
        "errors": errors,
    }


def split_reviewed_records(
    records: List[HumanReviewItem],
    train_ratio: float = 0.85,
    random_seed: int = 42,
) -> Tuple[List[HumanReviewItem], List[HumanReviewItem]]:
    """Split reviewed dataset records into category-balanced, seed-grouped train/validation sets.

    CRITICAL INVARIANT: Zero seed-family leakage. All records belonging to the same
    seed_id (e.g. coding_001) are strictly assigned to either Train or Validation.
    Categories are balanced across splits.
    """
    if not records:
        return [], []

    rng = random.Random(random_seed)

    # 1. Group records by category and then by seed_id
    cat_to_seed_groups: Dict[str, Dict[str, List[HumanReviewItem]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        group_key = r.seed_id or f"unbound_{r.record_id}"
        cat_to_seed_groups[r.category][group_key].append(r)

    train_records: List[HumanReviewItem] = []
    val_records: List[HumanReviewItem] = []

    # 2. For each category, partition seed groups
    for cat in sorted(cat_to_seed_groups.keys()):
        seed_dict = cat_to_seed_groups[cat]
        seed_keys = sorted(list(seed_dict.keys()))
        rng.shuffle(seed_keys)

        n_seeds = len(seed_keys)
        if n_seeds <= 1:
            # Single seed in this category: allocate to train
            train_seeds = set(seed_keys)
            val_seeds = set()
        else:
            # Allocate at least 1 seed to validation if ratio permits
            n_val = max(1, int(round(n_seeds * (1.0 - train_ratio))))
            if n_val >= n_seeds:
                n_val = 1
            val_seeds = set(seed_keys[:n_val])
            train_seeds = set(seed_keys[n_val:])

        for sk, rec_list in seed_dict.items():
            if sk in train_seeds:
                train_records.extend(rec_list)
            else:
                val_records.extend(rec_list)

    return train_records, val_records


def generate_review_report(queue_path: str) -> Dict[str, Any]:
    """Generate comprehensive audit report of human reviews and provisional proposals."""
    items = load_review_queue(queue_path)

    total = len(items)
    pending_human = sum(1 for i in items if not i.verified_by_human or i.decision == HumanReviewDecision.PENDING)
    human_verified = sum(1 for i in items if i.verified_by_human and i.decision != HumanReviewDecision.PENDING)
    approved = sum(1 for i in items if i.verified_by_human and i.decision == HumanReviewDecision.APPROVE)
    revised = sum(1 for i in items if i.verified_by_human and i.decision == HumanReviewDecision.REVISE)
    rejected = sum(1 for i in items if i.verified_by_human and i.decision == HumanReviewDecision.REJECT)

    provisional_stats: Dict[str, int] = defaultdict(int)
    for i in items:
        if i.provisional_decision:
            provisional_stats[i.provisional_decision.value] += 1

    cat_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"APPROVE": 0, "REVISE": 0, "REJECT": 0, "PENDING": 0, "TOTAL": 0})
    rejection_reasons: Dict[str, int] = defaultdict(int)
    human_reviewers: Set[str] = set()

    for i in items:
        dec_val = i.decision.value if i.verified_by_human else "PENDING"
        cat_stats[i.category][dec_val] += 1
        cat_stats[i.category]["TOTAL"] += 1
        if i.verified_by_human and i.decision == HumanReviewDecision.REJECT and i.reason:
            rejection_reasons[i.reason] += 1
        if i.verified_by_human and i.reviewer:
            human_reviewers.add(i.reviewer)

    return {
        "total_records": total,
        "pending_human_approvals": pending_human,
        "human_verified_count": human_verified,
        "human_approved_count": approved,
        "human_revised_count": revised,
        "human_rejected_count": rejected,
        "usable_sft_count": approved + revised,
        "provisional_proposals": dict(provisional_stats),
        "category_distribution": dict(cat_stats),
        "rejection_reasons": dict(rejection_reasons),
        "human_reviewers": sorted(list(human_reviewers)),
    }


def export_human_reviewed_dataset(
    queue_path: str,
    output_dir: str,
    dataset_version: str,
    train_ratio: float = 0.85,
    random_seed: int = 42,
    strict: bool = True,
) -> Dict[str, Any]:
    """Export the validated, human-approved dataset.

    STRICT GOVERNANCE INVARIANTS:
    1. Unreviewed (PENDING or unverified) records are NEVER admitted into final training exports.
    2. SFT exports include only verified human APPROVE and REVISE decisions.
    3. Revised targets are validated before admission.
    4. Clarification pairs are separated from SFT pairs.
    5. Zero seed-family leakage between Train and Validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    items = load_review_queue(queue_path)

    pending_items = [i for i in items if not i.verified_by_human or i.decision == HumanReviewDecision.PENDING]
    if strict and pending_items:
        raise ValueError(
            f"Cannot export dataset: {len(pending_items)} records are still PENDING human verification."
        )

    # Filter admitted items: only verified human APPROVE or REVISE
    admitted_items = [
        i for i in items
        if i.verified_by_human and i.decision in (HumanReviewDecision.APPROVE, HumanReviewDecision.REVISE)
    ]
    rejected_items = [i for i in items if i.verified_by_human and i.decision == HumanReviewDecision.REJECT]

    # Separate SFT positive enhancement from clarification pairs
    sft_items = [i for i in admitted_items if not i.clarification_required]
    clarification_items = [i for i in admitted_items if i.clarification_required]

    # Category-balanced, seed-grouped split for SFT
    sft_train, sft_val = split_reviewed_records(sft_items, train_ratio=train_ratio, random_seed=random_seed)

    # Category-balanced split for clarification if any exist
    clar_train, clar_val = split_reviewed_records(clarification_items, train_ratio=train_ratio, random_seed=random_seed)

    # 1. Export SFT Train & Validation
    def _write_sft(items_to_write: List[HumanReviewItem], filename: str) -> str:
        p = os.path.join(output_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            for it in items_to_write:
                target = it.final_target
                ex = SFTExample(
                    messages=[
                        SFTMessage(role="user", content=it.original_prompt),
                        SFTMessage(role="assistant", content=target or ""),
                    ],
                    metadata={
                        "record_id": it.record_id,
                        "category": it.category,
                        "seed_id": it.seed_id,
                        "quality_score": it.quality_score,
                        "review_decision": it.decision.value,
                        "reviewed_by": it.reviewer,
                        "review_reason": it.reason,
                        "reviewed_at": it.reviewed_at,
                        "original_target_preserved": it.teacher_target,
                        "target_revised": (it.decision == HumanReviewDecision.REVISE),
                        "provenance": it.metadata.get("provenance", {}),
                    },
                )
                f.write(ex.model_dump_json() + "\n")
        return p

    sft_train_path = _write_sft(sft_train, "sft_train.jsonl")
    sft_val_path = _write_sft(sft_val, "sft_validation.jsonl")

    # 2. Export Clarification Train & Validation
    def _write_clar(items_to_write: List[HumanReviewItem], filename: str) -> str:
        p = os.path.join(output_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            for it in items_to_write:
                target = it.final_target or (
                    "To provide the best enhancement, could you please clarify the following:\n"
                    + "\n".join(f"- {q}" for q in it.clarification_questions)
                )
                ex = ClarificationExample(
                    messages=[
                        SFTMessage(role="user", content=it.original_prompt),
                        SFTMessage(role="assistant", content=target),
                    ],
                    metadata={
                        "record_id": it.record_id,
                        "category": it.category,
                        "seed_id": it.seed_id,
                        "review_decision": it.decision.value,
                        "reviewed_by": it.reviewer,
                        "review_reason": it.reason,
                        "reviewed_at": it.reviewed_at,
                    },
                )
                f.write(ex.model_dump_json() + "\n")
        return p

    clar_train_path = _write_clar(clar_train, "clarification_train.jsonl")
    clar_val_path = _write_clar(clar_val, "clarification_validation.jsonl")

    # 3. Export Rejected Records
    rejected_path = os.path.join(output_dir, "rejected_records.jsonl")
    with open(rejected_path, "w", encoding="utf-8") as f:
        for it in rejected_items:
            entry = {
                "record_id": it.record_id,
                "category": it.category,
                "seed_id": it.seed_id,
                "original_prompt": it.original_prompt,
                "teacher_target": it.teacher_target,
                "rejection_reason": it.reason,
                "reviewed_by": it.reviewer,
                "reviewed_at": it.reviewed_at,
            }
            f.write(json.dumps(entry) + "\n")

    # 4. Export Complete Review Audit Trail
    audit_path = os.path.join(output_dir, "review_audit.jsonl")
    with open(audit_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")

    # 5. Build and Write Manifest & Dataset Card
    report = generate_review_report(queue_path)
    reviewers = report.get("human_reviewers", [])

    manifest = HumanReviewManifest(
        dataset_version=dataset_version,
        source_export_version=items[0].dataset_version if items else "unknown",
        total_reviewed=report["human_verified_count"],
        approved_count=report["human_approved_count"],
        revised_count=report["human_revised_count"],
        rejected_count=report["human_rejected_count"],
        pending_count=report["pending_human_approvals"],
        category_counts={cat: data["TOTAL"] for cat, data in report["category_distribution"].items()},
        rejection_reasons=report["rejection_reasons"],
        split_counts={
            "sft_train": len(sft_train),
            "sft_validation": len(sft_val),
            "clarification_train": len(clar_train),
            "clarification_validation": len(clar_val),
            "rejected": len(rejected_items),
        },
        reviewers=reviewers,
    )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    # Dataset Card
    card_path = os.path.join(output_dir, "DATASET_CARD.md")
    card_content = (
        f"# HumanIntent Human-Reviewed Dataset Card: {dataset_version}\n\n"
        f"**Source Build Version:** {manifest.source_export_version}  \n"
        f"**Reviewed At:** {manifest.reviewed_at}  \n"
        f"**Reviewers:** {', '.join(reviewers) if reviewers else 'human_auditor'}  \n\n"
        f"## Human Review Governance\n"
        f"Every example in this dataset has undergone individual expert human inspection.\n"
        f"- **No Unreviewed Records**: All training examples are strictly `APPROVE` (verified training-grade as generated) "
        f"or `REVISE` (human-corrected target validated against original prompt).\n"
        f"- **Provenance Integrity**: The original teacher target is preserved alongside any corrected target.\n"
        f"- **Split Isolation**: Train and Validation sets are grouped by `seed_id` to eliminate seed-family leakage.\n\n"
        f"## Dataset Statistics\n"
        f"- **Total Reviewed Records:** {manifest.total_reviewed}\n"
        f"- **Approved Unchanged:** {manifest.approved_count}\n"
        f"- **Revised & Corrected:** {manifest.revised_count}\n"
        f"- **Rejected Excluded:** {manifest.rejected_count}\n"
        f"- **Net Usable SFT Training Examples:** {manifest.approved_count + manifest.revised_count}\n\n"
        f"### Splits\n"
        f"- **SFT Train:** {manifest.split_counts.get('sft_train', 0)}\n"
        f"- **SFT Validation:** {manifest.split_counts.get('sft_validation', 0)}\n"
        f"- **Clarification Train:** {manifest.split_counts.get('clarification_train', 0)}\n"
        f"- **Clarification Validation:** {manifest.split_counts.get('clarification_validation', 0)}\n\n"
        f"### Category Coverage (Admitted SFT)\n"
        + "\n".join(f"- **{cat}:** {len([x for x in admitted_items if x.category == cat])}" for cat in sorted(set(x.category for x in admitted_items)))
        + f"\n\n### Rejection Reasons\n"
        + "\n".join(f"- `{reason}`: {cnt}" for reason, cnt in manifest.rejection_reasons.items())
        + "\n"
    )
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    return {
        "status": "SUCCESS",
        "output_dir": output_dir,
        "manifest": manifest.model_dump(),
        "report": report,
    }
