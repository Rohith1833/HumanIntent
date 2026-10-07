#!/usr/bin/env python3
"""
HumanIntent Holdout Benchmark Evaluation Runner.

This script executes holdout generalization evaluations from an external JSONL file
without contaminating the repository with implementation-visible holdout prompts.

Usage:
    python evaluation/run_holdout.py --file path/to/holdout.jsonl
    python evaluation/run_holdout.py --file path/to/holdout.jsonl --api-url http://localhost:8000/api/enhance
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure backend package can be imported if running direct in-process
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


async def evaluate_prompt_direct(prompt: str) -> Dict[str, Any]:
    """Run analysis and enhancement using local in-process HumanIntent pipeline."""
    from app.core.enhancer import prompt_enhancer

    analysis, enhanced_prompt = await prompt_enhancer.enhance(prompt)
    return {
        "analysis": analysis.model_dump(),
        "enhanced_prompt": enhanced_prompt,
    }


async def evaluate_prompt_http(prompt: str, api_url: str) -> Dict[str, Any]:
    """Run analysis and enhancement via running FastAPI backend."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(api_url, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()


def check_invariants(
    item: Dict[str, Any],
    result: Dict[str, Any],
) -> List[str]:
    """Verify optional invariants specified in the holdout item."""
    violations: List[str] = []
    analysis = result.get("analysis", {})
    enhanced = result.get("enhanced_prompt", "")

    expected_category = item.get("expected_category")
    if expected_category:
        actual_category = analysis.get("classification", {}).get("category")
        if actual_category != expected_category:
            violations.append(
                f"Category mismatch: expected {expected_category}, got {actual_category}"
            )

    combined_text = (
        enhanced
        + " "
        + " ".join(analysis.get("explicit_requirements", []))
        + " "
        + " ".join(analysis.get("constraints", []))
        + " "
        + " ".join(analysis.get("preferences", []))
    ).lower()

    must_contain = item.get("must_contain", [])
    for term in must_contain:
        if term.lower() not in combined_text:
            violations.append(f"Invariant violation: must contain '{term}'")

    must_not_contain = item.get("must_not_contain", [])
    for term in must_not_contain:
        if term.lower() in combined_text:
            violations.append(f"Invariant violation: must NOT contain '{term}'")

    return violations


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Run HumanIntent holdout evaluation from an external JSONL file."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the external holdout JSONL file",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Optional HTTP URL of running backend (e.g. http://localhost:8000/api/enhance). If omitted, evaluates in-process.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to output evaluation results JSONL file",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: Holdout file not found: {args.file}", file=sys.stderr)
        return 1

    records: List[Dict[str, Any]] = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as err:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {err}", file=sys.stderr)

    if not records:
        print(f"No test prompts found in {args.file}")
        return 0

    print(f"Loaded {len(records)} holdout prompts from {args.file}")
    print("=" * 60)

    total = len(records)
    passed_invariants = 0
    results_to_save: List[Dict[str, Any]] = []

    for i, item in enumerate(records, start=1):
        prompt = item.get("prompt", "")
        if not prompt:
            continue

        print(f"\n[{i}/{total}] Prompt: {prompt[:75]}{'...' if len(prompt) > 75 else ''}")

        try:
            if args.api_url:
                result = await evaluate_prompt_http(prompt, args.api_url)
            else:
                result = await evaluate_prompt_direct(prompt)
        except Exception as exc:
            print(f"  FAILED with error: {exc}")
            continue

        analysis = result.get("analysis", {})
        enhanced = result.get("enhanced_prompt", "")
        classification = analysis.get("classification", {})

        print(f"  Category:               {classification.get('category')} (confidence: {classification.get('confidence')})")
        print(f"  Objective:              {analysis.get('objective')}")
        print(f"  Explicit Requirements:  {len(analysis.get('explicit_requirements', []))} items -> {analysis.get('explicit_requirements')}")
        print(f"  Constraints:            {len(analysis.get('constraints', []))} items -> {analysis.get('constraints')}")
        print(f"  Preferences:            {len(analysis.get('preferences', []))} items -> {analysis.get('preferences')}")
        print(f"  Missing Context:        {len(analysis.get('missing_context', []))} items -> {analysis.get('missing_context')}")
        print(f"  Enhanced Length:        {len(enhanced)} chars")

        violations = check_invariants(item, result)
        if violations:
            print("  Invariant Failures:")
            for v in violations:
                print(f"    - {v}")
        else:
            passed_invariants += 1
            if item.get("expected_category") or item.get("must_contain") or item.get("must_not_contain"):
                print("  Invariants:             ALL PASSED")

        if args.output:
            results_to_save.append({
                "prompt": prompt,
                "input_invariants": {
                    "expected_category": item.get("expected_category"),
                    "must_contain": item.get("must_contain"),
                    "must_not_contain": item.get("must_not_contain"),
                },
                "analysis": analysis,
                "enhanced_prompt": enhanced,
                "violations": violations,
            })

    print("\n" + "=" * 60)
    print(f"Holdout Evaluation Complete: {passed_invariants}/{total} passed all specified invariants.")

    if args.output and results_to_save:
        with open(args.output, "w", encoding="utf-8") as out_f:
            for r in results_to_save:
                out_f.write(json.dumps(r) + "\n")
        print(f"Detailed results saved to {args.output}")

    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
