#!/usr/bin/env python3
"""Auditable Human Review CLI for HumanIntent Dataset Factory.

Enables offline human review, approval, revision, rejection, validation,
and export of training-grade SFT datasets while strictly distinguishing
provisional AI proposals from verified human decisions.
"""

import argparse
import json
import os
import sys
from typing import Optional

# Ensure project root and backend are in sys.path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "backend"))

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
    validate_review_target,
)
from dataset_factory.schemas import HumanReviewDecision


def cmd_export_queue(args):
    """Export candidates to a review queue."""
    count = export_review_queue(
        source_path_or_dir=args.source,
        output_queue_path=args.output,
        dataset_version=args.version,
        only_accepted=not args.all,
    )
    print(f"Exported {count} items to review queue: {args.output}")


def cmd_list(args):
    """List review queue items with status and summary."""
    items = load_review_queue(args.queue)
    if args.status:
        target_status = args.status.upper()
        if target_status == "PENDING":
            items = [i for i in items if not i.verified_by_human or i.decision.value == "PENDING"]
        else:
            items = [i for i in items if i.verified_by_human and i.decision.value == target_status]

    print(f"\nReview Queue ({len(items)} records): {args.queue}")
    print("-" * 110)
    print(f"{'Record ID':<12} {'Seed ID':<15} {'Category':<15} {'Human Status':<14} {'AI Proposal':<14} {'Reviewer':<16} {'Notes'}")
    print("-" * 110)
    for it in items:
        rid = it.record_id[:10] + ".." if len(it.record_id) > 10 else it.record_id
        sid = (it.seed_id or "-")[:14]
        cat = it.category[:14]
        h_status = it.decision.value if it.verified_by_human else "PENDING [AI]"
        ai_prop = (it.provisional_decision.value if it.provisional_decision else "-")[:13]
        rev = (it.reviewer or "-")[:15]
        note = (it.reason or it.proposal_reason or "")[:25]
        print(f"{rid:<12} {sid:<15} {cat:<15} {h_status:<14} {ai_prop:<14} {rev:<16} {note}")
    print("-" * 110)


def cmd_show(args):
    """Show detailed view of a single review item."""
    items = load_review_queue(args.queue)
    match = None
    for it in items:
        if it.record_id == args.record_id or it.record_id.startswith(args.record_id):
            match = it
            break

    if not match:
        print(f"Error: Record '{args.record_id}' not found in {args.queue}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"RECORD: {match.record_id}")
    print("=" * 80)
    print(f"Dataset Version     : {match.dataset_version}")
    print(f"Category            : {match.category}")
    print(f"Seed ID             : {match.seed_id}")
    print(f"Clarification Req.  : {match.clarification_required}")
    print("-" * 80)
    print("PROVISIONAL AI PROPOSAL:")
    print(f"  Suggested Decision: {match.provisional_decision.value if match.provisional_decision else 'None'}")
    print(f"  Proposal Source   : {match.proposal_source or 'None'}")
    print(f"  Proposal Reason   : {match.proposal_reason or 'None'}")
    print("-" * 80)
    print("HUMAN AUDITOR DECISION:")
    print(f"  Verified by Human : {match.verified_by_human}")
    print(f"  Final Decision    : {match.decision.value}")
    print(f"  Human Reviewer    : {match.reviewer or '[Awaiting Human Auditor]'}")
    print(f"  Reviewed At       : {match.reviewed_at or 'None'}")
    print(f"  Audit Reason      : {match.reason or 'None'}")
    print("-" * 80)
    print("ORIGINAL USER PROMPT:")
    print(match.original_prompt.strip())
    print("-" * 80)
    print("TEACHER ENHANCED TARGET:")
    print(match.teacher_target.strip() if match.teacher_target else "[None]")
    if match.proposed_target:
        print("-" * 80)
        print("AI PROPOSED REVISED TARGET:")
        print(match.proposed_target.strip())
    if match.revised_target:
        print("-" * 80)
        print("HUMAN VERIFIED REVISED TARGET:")
        print(match.revised_target.strip())
    print("=" * 80 + "\n")


def cmd_accept_proposal(args):
    """Human auditor accepts and certifies an AI provisional proposal."""
    item = accept_proposal(
        queue_path=args.queue,
        record_id=args.record_id,
        reviewer=args.reviewer,
        reason=args.reason,
        validate=not args.no_validate,
    )
    print(f"Human auditor '{item.reviewer}' ACCEPTED proposal {item.decision.value} for record {item.record_id}")


def cmd_approve(args):
    """Human auditor approves a record unchanged."""
    item = apply_review_decision(
        queue_path=args.queue,
        record_id=args.record_id,
        decision="APPROVE",
        reason=args.reason or "Verified training-grade",
        reviewer=args.reviewer,
    )
    print(f"Human auditor '{item.reviewer}' APPROVED record {item.record_id}")


def cmd_revise(args):
    """Human auditor revises a record with a custom target prompt."""
    target_content = None
    if args.target_file:
        with open(args.target_file, "r", encoding="utf-8") as f:
            target_content = f.read().strip()
    elif args.target:
        target_content = args.target.strip()
    else:
        print("Error: Either --target or --target-file must be provided for REVISE.", file=sys.stderr)
        sys.exit(1)

    if not args.reason:
        print("Error: --reason is required for REVISE.", file=sys.stderr)
        sys.exit(1)

    item = apply_review_decision(
        queue_path=args.queue,
        record_id=args.record_id,
        decision="REVISE",
        revised_target=target_content,
        reason=args.reason,
        reviewer=args.reviewer,
        validate=not args.no_validate,
    )
    print(f"Human auditor '{item.reviewer}' REVISED record {item.record_id}")


def cmd_reject(args):
    """Human auditor rejects and excludes a record."""
    if not args.reason:
        print("Error: --reason is required for REJECT.", file=sys.stderr)
        sys.exit(1)

    item = apply_review_decision(
        queue_path=args.queue,
        record_id=args.record_id,
        decision="REJECT",
        reason=args.reason,
        reviewer=args.reviewer,
    )
    print(f"Human auditor '{item.reviewer}' REJECTED record {item.record_id}")


def cmd_batch_propose(args):
    """Record batch provisional AI proposals (marks as PENDING human verification)."""
    with open(args.proposals_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "proposals" in data:
        proposals = data["proposals"]
    elif isinstance(data, dict) and "decisions" in data:
        proposals = data["decisions"]
    elif isinstance(data, list):
        proposals = data
    else:
        print("Error: Proposals file must be a JSON array or an object with 'proposals' list.", file=sys.stderr)
        sys.exit(1)

    result = batch_apply_proposals(
        queue_path=args.queue,
        proposals=proposals,
        proposal_source=args.source or "ai_provisional_reviewer",
        validate=not args.no_validate,
    )
    print(f"Recorded {result['applied_count']} of {result['total_specified']} provisional AI proposals.")
    if result["errors"]:
        print("\nErrors encountered:")
        for err in result["errors"]:
            print(f"  - {err}")


def cmd_report(args):
    """Print review report."""
    rep = generate_review_report(args.queue)
    print("\n" + "=" * 60)
    print("HUMAN REVIEW AUDIT REPORT")
    print("=" * 60)
    print(f"Total Records in Queue        : {rep['total_records']}")
    print(f"Pending Human Approvals       : {rep['pending_human_approvals']}")
    print(f"Human Verified Decisions      : {rep['human_verified_count']}")
    print(f"  - Human Approved            : {rep['human_approved_count']}")
    print(f"  - Human Revised             : {rep['human_revised_count']}")
    print(f"  - Human Rejected            : {rep['human_rejected_count']}")
    print(f"Admissible SFT Training Yield : {rep['usable_sft_count']}")
    print("-" * 60)
    print("Provisional AI Proposals Summary (Unverified):")
    for dec_name, count in sorted(rep["provisional_proposals"].items()):
        print(f"  - Suggested {dec_name:<10}: {count}")
    print("-" * 60)
    print("Human Decision Category Breakdown:")
    for cat, stats in sorted(rep["category_distribution"].items()):
        print(f"  {cat:<18}: App={stats['APPROVE']} Rev={stats['REVISE']} Rej={stats['REJECT']} Pend={stats['PENDING']} (Total: {stats['TOTAL']})")
    print("-" * 60)
    print("Human Rejection Reasons:")
    if rep["rejection_reasons"]:
        for reason, cnt in sorted(rep["rejection_reasons"].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {cnt}")
    else:
        print("  (None verified by human)")
    print("-" * 60)
    print(f"Verified Human Reviewers: {', '.join(rep['human_reviewers']) if rep['human_reviewers'] else 'None (All Pending Human Review)'}")
    print("=" * 60 + "\n")


def cmd_export_dataset(args):
    """Export verified human-approved dataset."""
    result = export_human_reviewed_dataset(
        queue_path=args.queue,
        output_dir=args.output_dir,
        dataset_version=args.version,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        strict=not args.allow_pending,
    )
    print(f"\nSuccessfully exported human-reviewed dataset to: {args.output_dir}")
    print(f"Manifest written to: {os.path.join(args.output_dir, 'manifest.json')}")
    print(f"Dataset card written to: {os.path.join(args.output_dir, 'DATASET_CARD.md')}")
    split_counts = result["manifest"]["split_counts"]
    print("Split Summary:")
    for split_name, cnt in split_counts.items():
        print(f"  - {split_name}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Auditable Human Review Workflow for HumanIntent Datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export-queue
    p_eq = subparsers.add_parser("export-queue", help="Export candidate records to human review queue")
    p_eq.add_argument("--source", required=True, help="Path to candidates.jsonl or export directory")
    p_eq.add_argument("--output", required=True, help="Path to output review queue JSONL")
    p_eq.add_argument("--version", required=True, help="Dataset version identifier")
    p_eq.add_argument("--all", action="store_true", help="Include non-accepted candidates (when reading candidates.jsonl)")
    p_eq.set_defaults(func=cmd_export_queue)

    # list
    p_ls = subparsers.add_parser("list", help="List review items in queue")
    p_ls.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_ls.add_argument("--status", choices=["PENDING", "APPROVE", "REVISE", "REJECT"], help="Filter by review status")
    p_ls.set_defaults(func=cmd_list)

    # show
    p_sh = subparsers.add_parser("show", help="Show full content of a review item")
    p_sh.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_sh.add_argument("--record-id", required=True, help="Record ID or ID prefix")
    p_sh.set_defaults(func=cmd_show)

    # accept-proposal
    p_ac = subparsers.add_parser("accept-proposal", help="Human auditor accepts an AI provisional proposal")
    p_ac.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_ac.add_argument("--record-id", required=True, help="Record ID or prefix")
    p_ac.add_argument("--reviewer", required=True, help="Human reviewer name or ID (Required)")
    p_ac.add_argument("--reason", help="Optional override reason")
    p_ac.add_argument("--no-validate", action="store_true", help="Skip curation validation on proposed target")
    p_ac.set_defaults(func=cmd_accept_proposal)

    # approve
    p_ap = subparsers.add_parser("approve", help="Human auditor approves record unchanged")
    p_ap.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_ap.add_argument("--record-id", required=True, help="Record ID or prefix")
    p_ap.add_argument("--reviewer", required=True, help="Human reviewer name or ID (Required)")
    p_ap.add_argument("--reason", default="Verified training-grade", help="Approval notes")
    p_ap.set_defaults(func=cmd_approve)

    # revise
    p_rv = subparsers.add_parser("revise", help="Human auditor revises record with custom target")
    p_rv.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_rv.add_argument("--record-id", required=True, help="Record ID or prefix")
    p_rv.add_argument("--reviewer", required=True, help="Human reviewer name or ID (Required)")
    p_rv.add_argument("--target", help="Revised prompt target text")
    p_rv.add_argument("--target-file", help="Path to file containing revised prompt target")
    p_rv.add_argument("--reason", required=True, help="Revision rationale / defect description")
    p_rv.add_argument("--no-validate", action="store_true", help="Skip curation validation on revised target")
    p_rv.set_defaults(func=cmd_revise)

    # reject
    p_rj = subparsers.add_parser("reject", help="Human auditor rejects and excludes record")
    p_rj.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_rj.add_argument("--record-id", required=True, help="Record ID or prefix")
    p_rj.add_argument("--reviewer", required=True, help="Human reviewer name or ID (Required)")
    p_rj.add_argument("--reason", required=True, help="Rejection rationale / defect reason")
    p_rj.set_defaults(func=cmd_reject)

    # batch-propose
    p_bp = subparsers.add_parser("batch-propose", help="Record batch provisional AI proposals (marks as PENDING human verification)")
    p_bp.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_bp.add_argument("--proposals-file", required=True, help="Path to JSON file with proposals array")
    p_bp.add_argument("--source", default="ai_provisional_reviewer", help="AI proposal source identifier")
    p_bp.add_argument("--no-validate", action="store_true", help="Skip curation validation on proposed targets")
    p_bp.set_defaults(func=cmd_batch_propose)

    # report
    p_rp = subparsers.add_parser("report", help="Display human review audit report")
    p_rp.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_rp.set_defaults(func=cmd_report)

    # export-dataset
    p_ed = subparsers.add_parser("export-dataset", help="Export validated human-reviewed dataset")
    p_ed.add_argument("--queue", required=True, help="Path to review queue JSONL")
    p_ed.add_argument("--output-dir", required=True, help="Output export directory")
    p_ed.add_argument("--version", required=True, help="Final dataset version name")
    p_ed.add_argument("--train-ratio", type=float, default=0.85, help="Train split ratio (default 0.85)")
    p_ed.add_argument("--seed", type=int, default=42, help="Random seed for grouping split (default 42)")
    p_ed.add_argument("--allow-pending", action="store_true", help="Allow unreviewed items (excluded from export)")
    p_ed.set_defaults(func=cmd_export_dataset)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
