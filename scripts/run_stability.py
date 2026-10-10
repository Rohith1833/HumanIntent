#!/usr/bin/env python3
"""
Run stochastic stability evaluation for Phase 10 (TASK 20).
Executes at least 30 frozen benchmark cases across 3 repetitions.
Reports category stability, clarification stability, and quality score variance.
"""

import argparse
import asyncio
import json
import logging
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(repo_root, "backend")
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.benchmark_loader import load_benchmark
from evaluation.stability import run_stability_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_stability")


async def main():
    parser = argparse.ArgumentParser(description="Phase 10 Stability Runner")
    parser.add_argument("--benchmark", default=os.path.join("benchmarks", "v1", "benchmark.jsonl"))
    parser.add_argument("--metadata", default=os.path.join("benchmarks", "v1", "metadata.json"))
    parser.add_argument("--cases", type=int, default=30, help="Number of benchmark cases to evaluate (>= 30)")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per case (default: 3)")
    parser.add_argument("--model", default="llama3:latest")
    parser.add_argument("--output", default=os.path.join("evaluation_results", "full_v1", "stability.json"))
    args = parser.parse_args()

    logger.info("Loading benchmark cases for stability analysis...")
    cases = load_benchmark(
        benchmark_path=args.benchmark,
        metadata_path=args.metadata,
        limit=args.cases,
    )
    logger.info("Loaded %d benchmark cases. Executing %d repetitions each...", len(cases), args.repetitions)

    metrics = await run_stability_evaluation(cases=cases, repetitions=args.repetitions, model=args.model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(metrics.model_dump_json(indent=2))

    print("\n" + "=" * 60)
    print("PHASE 10 STABILITY EVALUATION RESULTS (TASK 20)")
    print("=" * 60)
    print(f"Total Cases Evaluated:        {metrics.total_cases_evaluated}")
    print(f"Repetitions per Case:         {metrics.repetitions_per_case}")
    print(f"Category Stability Rate:      {metrics.category_stability_rate:.1%}")
    print(f"Clarification Stability Rate: {metrics.clarification_stability_rate:.1%}")
    print(f"Mean Quality Score Std:       {metrics.mean_quality_score_std:.4f}")
    print(f"Max Quality Score Std:        {metrics.max_quality_score_std:.4f}")
    print(f"Saved To:                     {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
