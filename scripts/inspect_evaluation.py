#!/usr/bin/env python3
"""
Inspect and summarize Phase 10 evaluation runs.

Usage:
    python scripts/inspect_evaluation.py --run-dir evaluation_results/run_001
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Inspect Phase 10 evaluation run artifacts.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=os.path.join("evaluation_results", "run_001"),
        help="Path to evaluation run output directory",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print top error analysis cases",
    )
    args = parser.parse_args()

    metrics_path = os.path.join(args.run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Error: metrics.json not found in {args.run_dir}")
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("\n" + "=" * 60)
    print(f"PHASE 10 EVALUATION RUN: {metrics.get('run_id')}")
    print(f"Benchmark Version:        {metrics.get('benchmark_version')}")
    print(f"Model:                    {metrics.get('model')}")
    print(f"Total Cases:              {metrics.get('total_cases')}")
    print("=" * 60)

    print("\n[RELEASE GATES STATUS]")
    for gate in metrics.get("release_gates", []):
        icon = "[PASS]" if gate.get("status") == "PASSED" else "[FAIL]"
        print(f"  {icon} {gate.get('gate_name'):<30} Target: {gate.get('target_threshold'):<10} Actual: {gate.get('actual_value')}%")

    print("\n[PRIMARY ACCURACY & INVARIANTS]")
    print(f"  Category Accuracy:          {metrics.get('category_accuracy', 0.0):.1%}")
    print(f"  Task-Family Accuracy:       {metrics.get('task_family_accuracy', 0.0):.1%}")
    print(f"  Requirement Recall:         {metrics.get('requirement_recall', 0.0):.1%}")
    print(f"  Constraint Recall:          {metrics.get('constraint_recall', 0.0):.1%}")
    print(f"  Quantity Preservation:      {metrics.get('quantity_preservation', 0.0):.1%}")
    print(f"  Unsupported Req Rate:       {metrics.get('unsupported_requirement_rate', 0.0):.1%}")
    print(f"  Clarification F1:           {metrics.get('clarification_f1', 0.0):.1%}")
    print(f"  Meta Leakage Rate:          {metrics.get('meta_leakage_rate', 0.0):.1%}")
    print(f"  Median Expansion:           {metrics.get('median_expansion_ratio', 1.0):.2f}x")

    print("\n[PAIRWISE WIN RATES]")
    print(f"  Win Rate vs. Raw:           {metrics.get('win_rate_vs_raw', 0.0):.1%}")
    print(f"  Win Rate vs. Generic:       {metrics.get('win_rate_vs_generic', 0.0):.1%}")

    print("\n[TOP FAILURE CLASSES]")
    failures = metrics.get("failure_counts", {})
    if failures:
        for k, v in sorted(failures.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {k}: {v}")
    else:
        print("  None detected.")

    if args.show_failures:
        err_path = os.path.join(args.run_dir, "error_analysis.jsonl")
        if os.path.exists(err_path):
            print("\n[SAMPLE ERROR CASES]")
            with open(err_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if idx > 5:
                        break
                    err = json.loads(line)
                    print(f"\nCase #{idx}: {err.get('case_id')} ({err.get('category')})")
                    print(f"  Failures: {err.get('failure_types')}")
                    print(f"  Prompt:   {err.get('user_prompt')[:80]}...")
                    print(f"  Output:   {err.get('enhanced_output')[:80]}...")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
