#!/usr/bin/env python3
"""
HumanIntent Dataset Inspector CLI.

Inspects, validates, and summarizes a generated dataset export directory.

Usage:
    python scripts/inspect_dataset.py --dir datasets/exports/humanintent-v0.1.0
    python scripts/inspect_dataset.py --version humanintent-v0.1.0
"""

import argparse
import json
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def count_lines(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def inspect_dataset(export_dir: str):
    print(f"\n=======================================================")
    print(f"INSPECTING DATASET: {export_dir}")
    print(f"=======================================================")

    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest.json not found in {export_dir}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Dataset Version:    {manifest.get('version')}")
    print(f"Created At:         {manifest.get('created_at')}")
    print(f"Factory Version:    {manifest.get('factory_version')}")
    print(f"Generator Model:    {manifest.get('generator_model')}")
    print(f"Random Seed:        {manifest.get('random_seed')}")

    rec_counts = manifest.get("record_counts", {})
    print(f"\n--- RECORD COUNTS ---")
    print(f"Total Candidates:   {rec_counts.get('total', 0)}")
    print(f"Accepted Records:   {rec_counts.get('accepted', 0)}")
    print(f"Rejected Records:   {rec_counts.get('rejected', 0)}")
    print(f"Sanitized Count:    {manifest.get('sanitization_count', 0)}")
    print(f"Deduped Duplicates: {manifest.get('dedup_count', 0)}")

    print(f"\n--- EXPORT ARTIFACTS & ROW COUNTS ---")
    artifacts = [
        "sft_train.jsonl",
        "sft_validation.jsonl",
        "clarification_train.jsonl",
        "clarification_validation.jsonl",
        "preference_train.jsonl",
        "preference_validation.jsonl",
        "review_sample.jsonl",
        "DATASET_CARD.md",
    ]
    for art in artifacts:
        p = os.path.join(export_dir, art)
        if os.path.exists(p):
            lines = count_lines(p) if art.endswith(".jsonl") else "PRESENT"
            print(f"  [+] {art:<30}: {lines}")
        else:
            print(f"  [-] {art:<30}: MISSING")

    print(f"\n--- CATEGORY DISTRIBUTION (ACCEPTED) ---")
    for cat, cnt in manifest.get("category_counts", {}).items():
        print(f"  - {cat:<20}: {cnt}")

    qd = manifest.get("quality_distribution", {})
    if qd:
        print(f"\n--- QUALITY DISTRIBUTION ---")
        print(f"  Mean Score:       {qd.get('mean', 0.0)}/100")
        print(f"  Min Score:        {qd.get('min', 0.0)}/100")
        print(f"  Max Score:        {qd.get('max', 0.0)}/100")
        print(f"  Evaluated Count:  {qd.get('sample_size', 0)}")

    splits = manifest.get("split_counts", {})
    print(f"\n--- SPLIT COUNTS ---")
    print(f"  Train:            {splits.get('train', 0)}")
    print(f"  Validation:       {splits.get('validation', 0)}")

    # Inspect a few sample review records
    review_path = os.path.join(export_dir, "review_sample.jsonl")
    if os.path.exists(review_path):
        print(f"\n--- SAMPLE REVIEW RECORDS (FIRST 2) ---")
        with open(review_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i > 2:
                    break
                item = json.loads(line)
                print(f"[{i}] Category: {item.get('category')} | Source: {item.get('source_type')}")
                print(f"    Input:    {item.get('original_prompt')[:80]}...")
                enhanced = item.get('enhanced_prompt')
                if enhanced:
                    print(f"    Enhanced: {enhanced[:80]}...")
                print(f"    Quality:  {item.get('quality_score')} | Clarification: {item.get('clarification_required')}")

    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect a HumanIntent dataset export package.")
    parser.add_argument("--dir", type=str, default=None, help="Direct path to dataset export directory")
    parser.add_argument("--version", type=str, default=None, help="Version tag under datasets/exports/")
    args = parser.parse_args()

    if args.dir:
        target_dir = args.dir
    elif args.version:
        target_dir = os.path.join("datasets", "exports", args.version)
    else:
        # Default to latest export under datasets/exports
        base = os.path.join("datasets", "exports")
        if os.path.exists(base):
            subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
            if subdirs:
                target_dir = sorted(subdirs)[-1]
            else:
                print("No exports found under datasets/exports.")
                sys.exit(1)
        else:
            print("datasets/exports directory does not exist.")
            sys.exit(1)

    inspect_dataset(target_dir)


if __name__ == "__main__":
    main()
