#!/usr/bin/env python3
"""
HumanIntent Phase 10 Evaluation CLI.

Runs formal, reproducible evaluations of HumanIntent and baselines
against the frozen benchmark.

Usage:
    python scripts/run_evaluation.py --dry-run
    python scripts/run_evaluation.py --limit 20 --output evaluation_results/pilot_001
    python scripts/run_evaluation.py --benchmark benchmarks/v1/benchmark.jsonl
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

from evaluation.config import EvaluationConfig
from evaluation.runner import EvaluationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_evaluation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HumanIntent Phase 10 Benchmark Evaluation Runner.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=os.path.join("benchmarks", "v1", "benchmark.jsonl"),
        help="Path to frozen benchmark JSONL file",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=os.path.join("benchmarks", "v1", "metadata.json"),
        help="Path to benchmark metadata with SHA-256 digest",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3:latest",
        help="Ollama model tag to evaluate (default: llama3:latest)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="llama3:latest",
        help="Ollama model tag for independent semantic judge",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="raw,generic,humanintent",
        help="Comma-separated list of baselines: raw,generic,no_validator_ablation,humanintent",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("evaluation_results", "run_001"),
        help="Output directory for predictions and reports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of benchmark cases evaluated",
    )
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=None,
        help="Sample N cases per category across all categories",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated category filter (e.g. CODING,DEBUGGING)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Filter by difficulty: EASY, MEDIUM, or HARD",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of evaluation repetitions for stability analysis",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate benchmark integrity, schemas, and case counts without LLM calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute predictions ignoring existing checkpoints",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    cats = [c.strip().upper() for c in args.categories.split(",")] if args.categories else None
    baselines = [b.strip() for b in args.baselines.split(",")]

    config = EvaluationConfig(
        benchmark_path=args.benchmark,
        metadata_path=args.metadata,
        output_dir=args.output,
        model=args.model,
        judge_model=args.judge_model,
        baselines=baselines,
        limit=args.limit,
        sample_per_category=args.sample_per_category,
        categories=cats,
        difficulty=args.difficulty,
        repeat=args.repeat,
        dry_run=args.dry_run,
        force=args.force,
    )

    runner = EvaluationRunner(config)
    result = await runner.run()

    if args.dry_run:
        print("\n" + "=" * 55)
        print("PHASE 10 EVALUATION DRY RUN COMPLETE")
        print("=" * 55)
        print(f"Benchmark Hash Verified: {result.get('benchmark_verified')}")
        print(f"Eligible Cases:          {result.get('total_cases')}")
        print(f"Active Baselines:        {', '.join(result.get('baselines', []))}")
        print(f"Model:                   {result.get('model')}")
        print("Zero Ollama LLM calls made.")
        print("=" * 55 + "\n")
        return

    m = result.get("metrics", {})
    print("\n" + "=" * 55)
    print("PHASE 10 EVALUATION COMPLETED")
    print("=" * 55)
    print(f"Run ID:                  {result.get('run_id')}")
    print(f"Cases Evaluated:         {m.get('total_cases')}")
    print(f"Category Accuracy:       {m.get('category_accuracy', 0.0):.1%}")
    print(f"Task-Family Accuracy:    {m.get('task_family_accuracy', 0.0):.1%}")
    print(f"Requirement Recall:      {m.get('requirement_recall', 0.0):.1%}")
    print(f"Constraint Recall:       {m.get('constraint_recall', 0.0):.1%}")
    print(f"Quantity Preservation:   {m.get('quantity_preservation', 0.0):.1%}")
    print(f"Unsupported Req Rate:    {m.get('unsupported_requirement_rate', 0.0):.1%}")
    print(f"Clarification F1:        {m.get('clarification_f1', 0.0):.1%}")
    print(f"Pairwise Win vs Raw:     {result.get('pairwise', {}).get('vs_raw_win', 0.0):.1%}")
    print(f"Pairwise Win vs Generic: {result.get('pairwise', {}).get('vs_gen_win', 0.0):.1%}")
    print(f"Report Generated:        {result.get('report_path')}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
