#!/usr/bin/env python3
"""
HumanIntent Dataset Factory CLI.

Generates reproducible, versioned training datasets (SFT, Clarification, Preference)
from synthetic seed intents and consented real-user interactions.

Usage:
    python scripts/generate_dataset.py --version humanintent-v0.1.0 --seed datasets/seeds/v1.jsonl
    python scripts/generate_dataset.py --dry-run
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure repository root and backend are on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(repo_root, "backend")
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_factory.config import load_config
from dataset_factory.pipeline import DatasetFactoryPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HumanIntent Dataset Factory — Build versioned training datasets.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="humanintent-v0.1.0",
        help="Semantic dataset version tag (default: humanintent-v0.1.0)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="datasets/seeds/v1.jsonl",
        help="Path to seed intents JSONL file (default: datasets/seeds/v1.jsonl)",
    )
    parser.add_argument(
        "--synthetic-per-seed",
        type=int,
        default=3,
        help="Number of synthetic variants to generate per seed (default: 3)",
    )
    parser.add_argument(
        "--include-consented-real",
        action="store_true",
        help="Ingest consented real-user interactions from Phase 8 history",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        default=True,
        help="Build exclusively from synthetic seed variants (default: True)",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Build exclusively from consented real history records",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show planned candidate counts without generating",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Global random seed for deterministic sampling and splitting (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/exports",
        help="Destination directory for exported dataset packages (default: datasets/exports)",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="llama3:latest",
        help="Ollama model for synthetic generation (default: llama3:latest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML or JSON config file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration ignoring existing intermediate checkpoints and caches",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    if args.generator_model:
        os.environ["OLLAMA_MODEL"] = args.generator_model

    # Build config merging file config and CLI flags
    overrides = {
        "dataset_version": args.version,
        "seeds_path": args.seed,
        "synthetic_per_seed": args.synthetic_per_seed,
        "include_consented_real": args.include_consented_real,
        "synthetic_only": not args.include_consented_real if not args.real_only else False,
        "real_only": args.real_only,
        "dry_run": args.dry_run,
        "random_seed": args.random_seed,
        "output_dir": args.output_dir,
        "generator_model": args.generator_model,
        "force": args.force,
    }

    config = load_config(config_path=args.config, **overrides)
    pipeline = DatasetFactoryPipeline(config=config)

    result = await pipeline.run()

    if config.dry_run:
        print("\n=======================================================")
        print("DRY RUN SUMMARY")
        print("=======================================================")
        print(f"Dataset Version:             {result['dataset_version']}")
        print(f"Seed Count:                  {result['seed_count']}")
        print(f"Planned Synthetic Variants:  {result['planned_synthetic_candidates']}")
        print("Category Coverage:")
        for cat, count in result.get("category_distribution", {}).items():
            print(f"  - {cat:<20}: {count}")
        print("Dry run completed successfully. No files written.")
        print("=======================================================\n")
    else:
        stats = result.get("stats", {})
        print("\n=======================================================")
        print("DATASET FACTORY BUILD SUCCESSFUL")
        print("=======================================================")
        print(f"Dataset Version:        {result['dataset_version']}")
        print(f"Exports Directory:      {result['export_dir']}")
        print(f"Total Candidates:       {stats.get('total_candidates', 0)}")
        print(f"Accepted Records:       {stats.get('accepted_count', 0)}")
        print(f"Rejected Records:       {stats.get('rejected_count', 0)}")
        print(f"Acceptance Rate:        {stats.get('acceptance_rate', 0.0):.1%}")
        print(f"SFT Train Export:       {result['sft_train_path']}")
        print(f"SFT Val Export:         {result['sft_val_path']}")
        print(f"Clarification Export:   {result['clarification_train_path']}")
        print(f"Preference Export:      {result['preference_train_path']}")
        print(f"Review Sample:          {result['review_sample_path']}")
        print(f"Manifest File:          {result['manifest_path']}")
        print(f"Dataset Card:           {result['dataset_card_path']}")
        print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
