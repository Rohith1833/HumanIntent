import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from dataset_factory.curation import curate_candidate
from dataset_factory.schemas import DatasetRecord
from dataset_factory.seed_loader import load_seeds

def main():
    print("Running Dataset Curation Audit on humanintent-v0.1.0-live...")

    v010_dir = os.path.join("datasets", "exports", "humanintent-v0.1.0-live")
    train_path = os.path.join(v010_dir, "sft_train.jsonl")
    val_path = os.path.join(v010_dir, "sft_validation.jsonl")
    seeds_path = os.path.join("datasets", "seeds", "small_live_seeds.jsonl")

    seeds = load_seeds(seeds_path) if os.path.exists(seeds_path) else []
    seed_map = {s.seed_id: s for s in seeds}

    accepted_records: List[Dict[str, Any]] = []
    for path in [train_path, val_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        accepted_records.append(json.loads(line))

    # Also load intermediate candidates to correlate seed provenance
    inter_path = os.path.join("datasets", "intermediate", "humanintent-v0.1.0-live_candidates.jsonl")
    cand_map = {}
    if os.path.exists(inter_path):
        with open(inter_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = DatasetRecord.model_validate_json(line)
                    cand_map[rec.record_id] = rec

    audit_records = []
    passed_count = 0
    rejected_count = 0
    issue_counts: Dict[str, int] = {}

    for item in accepted_records:
        meta = item.get("metadata", {})
        rec_id = meta.get("record_id", "")
        category = meta.get("category", "")
        quality_score = meta.get("quality_score", 0.0)

        user_msg = next((m["content"] for m in item.get("messages", []) if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in item.get("messages", []) if m["role"] == "assistant"), "")

        cand_rec = cand_map.get(rec_id)
        seed_id = cand_rec.provenance.seed_id if cand_rec and cand_rec.provenance else None
        seed = seed_map.get(seed_id) if seed_id else None
        already_good = cand_rec.labels.already_good if cand_rec else False
        norm_intent = cand_rec.normalized_intent if cand_rec else None

        report = curate_candidate(
            original_prompt=user_msg,
            target_prompt=asst_msg,
            category=category,
            normalized_intent=norm_intent,
            seed=seed,
            clarification_required=False,
            already_good=already_good,
        )

        status = "PASS" if report.is_acceptable else "FAIL"
        if report.is_acceptable:
            passed_count += 1
        else:
            rejected_count += 1
            for issue in report.issues:
                t = issue.type.value
                issue_counts[t] = issue_counts.get(t, 0) + 1

        audit_records.append({
            "record_id": rec_id,
            "category": category,
            "quality_score": quality_score,
            "original_prompt": user_msg,
            "status": status,
            "is_acceptable": report.is_acceptable,
            "issues": [i.model_dump() for i in report.issues],
        })

    audit_result = {
        "dataset_version": "humanintent-v0.1.0-live",
        "total_previously_accepted": len(accepted_records),
        "now_passed": passed_count,
        "now_rejected": rejected_count,
        "rejection_rate": round(rejected_count / max(len(accepted_records), 1), 4),
        "issue_counts": issue_counts,
        "records": audit_records,
    }

    out_path = os.path.join(v010_dir, "curation_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    print(f"Audit completed: {passed_count} PASS, {rejected_count} FAIL out of {len(accepted_records)}.")
    print("Issues detected:")
    for k, v in issue_counts.items():
        print(f"  {k}: {v}")
    print(f"Audit written to: {out_path}")

if __name__ == "__main__":
    main()
