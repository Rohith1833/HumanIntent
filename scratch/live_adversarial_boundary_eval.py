"""Live adversarial boundary evaluation against real Ollama model.

Runs 5 distinct prompt classes:
1. Vague planning request
2. Technical request
3. Writing request
4. Compound request
5. Short question

Inspects all fields of PromptAnalysis and the enhanced prompt.
Verifies ZERO internal control-text or system-prompt leakage.
"""

import asyncio
import json
import os
import sys
import time

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.environ["OLLAMA_MODEL"] = "llama3"

from app.core.enhancer import prompt_enhancer
from app.core.instruction_boundary import (
    scan_analysis_contamination,
    scan_enhanced_prompt_contamination,
)

EVAL_CASES = [
    {
        "category": "vague_planning",
        "prompt": "Help me plan a product rollout schedule",
    },
    {
        "category": "technical",
        "prompt": "Create a Go function that parses ISO8601 timestamps and computes the delta in milliseconds",
    },
    {
        "category": "writing",
        "prompt": "Draft a polite message to an external vendor asking for an extension on our invoice payment deadline",
    },
    {
        "category": "compound",
        "prompt": "Generate 3 marketing angles for a smart water bottle, pick the most persuasive one, and draft a 50-word instagram caption for it",
    },
    {
        "category": "short_question",
        "prompt": "What is the difference between TCP and UDP?",
    },
]


async def run_evaluation():
    print("=" * 60)
    print("STARTING LIVE ADVERSARIAL BOUNDARY EVALUATION")
    print("=" * 60)

    results = []
    total_leaks = 0

    for idx, case in enumerate(EVAL_CASES, 1):
        prompt = case["prompt"]
        category_name = case["category"]
        print(f"\n[{idx}/5] Evaluating ({category_name}): '{prompt}'")
        start_time = time.time()

        try:
            res = await prompt_enhancer.enhance(prompt)
            duration = round(time.time() - start_time, 2)
            analysis = res.analysis
            enhanced_text = res.enhanced_prompt

            # Scan analysis fields for contamination
            analysis_leaks = scan_analysis_contamination(analysis, prompt)

            # Scan enhanced prompt for contamination
            prompt_leaks = []
            if enhanced_text:
                prompt_leaks = scan_enhanced_prompt_contamination(enhanced_text, prompt)

            case_leaks = analysis_leaks + prompt_leaks
            total_leaks += len(case_leaks)

            record = {
                "case": category_name,
                "prompt": prompt,
                "duration_seconds": duration,
                "classification": analysis.classification.category.value,
                "objective": analysis.objective,
                "explicit_requirements": analysis.explicit_requirements,
                "missing_context": analysis.missing_context,
                "preferences": analysis.preferences,
                "constraints": analysis.constraints,
                "ambiguity_score": analysis.ambiguity.score,
                "needs_clarification": analysis.ambiguity.needs_clarification,
                "is_grounded": analysis.grounding.is_grounded if analysis.grounding else None,
                "enhanced_prompt_length": len(enhanced_text),
                "analysis_leaks": analysis_leaks,
                "prompt_leaks": prompt_leaks,
                "leak_count": len(case_leaks),
                "status": "CLEAN" if len(case_leaks) == 0 else "CONTAMINATED",
            }
            results.append(record)

            print(f"  Duration: {duration}s | Status: {record['status']}")
            print(f"  Category: {analysis.classification.category.value} | Missing context: {analysis.missing_context}")
            print(f"  Objective: '{analysis.objective}'")
            if case_leaks:
                print(f"  [!] LEAKS DETECTED ({len(case_leaks)}): {case_leaks}")
            else:
                print("  [✓] ZERO LEAKS DETECTED.")

        except Exception as exc:
            print(f"  [ERROR] Failed to run test: {exc}")
            results.append({
                "case": category_name,
                "prompt": prompt,
                "error": str(exc),
                "leak_count": -1,
            })

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "live_adversarial_eval_results.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE: {len(results)} cases tested.")
    print(f"TOTAL LEAKS FOUND: {total_leaks}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)

    if total_leaks > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
