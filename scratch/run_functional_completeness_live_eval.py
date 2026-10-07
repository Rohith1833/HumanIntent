import asyncio
import json
import logging
import sys
sys.path.insert(0, ".")
from app.core.analyzer import PromptAnalyzer
from app.core.enhancer import PromptEnhancer
from app.core.validator import PromptValidator
from app.core.scorer import PromptScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_eval")

NEW_PROMPTS = [
    {
        "id": "1_artifact_type_only",
        "description": "Artifact type only",
        "prompt": "generate a report",
    },
    {
        "id": "2_artifact_platform_no_purpose",
        "description": "Artifact + platform but no functional purpose",
        "prompt": "build an iOS mobile app with Swift",
    },
    {
        "id": "3_artifact_plus_purpose",
        "description": "Artifact + functional purpose",
        "prompt": "create a presentation explaining photosynthesis to middle school students",
    },
    {
        "id": "4_short_complete_operation",
        "description": "Very short complete operation",
        "prompt": "compute the factorial of 10",
    },
    {
        "id": "5_detailed_complete_compound",
        "description": "Detailed complete compound request",
        "prompt": "Write a technical post comparing Docker vs Podman, include 3 security advantages of daemonless containers, and format as Markdown",
    },
]

async def run_live_eval():
    analyzer = PromptAnalyzer()
    enhancer = PromptEnhancer()
    validator = PromptValidator()
    scorer = PromptScorer()

    results = []

    for item in NEW_PROMPTS:
        p = item["prompt"]
        print(f"\n==========================================")
        print(f"EVALUATING: {item['id']} - '{p}'")
        print(f"==========================================")

        # 1. Analyze
        analysis = await analyzer.analyze(p)
        deliv = analysis.deliverable_semantics
        print(f"Deliverable Type: {deliv.deliverable_type if deliv else None}")
        print(f"Intended Outcome: {deliv.intended_outcome if deliv else None}")
        print(f"Is Functionally Complete: {deliv.is_functionally_complete if deliv else None}")
        print(f"Ambiguity Score: {analysis.ambiguity.score} ({analysis.ambiguity.level.value})")
        print(f"Needs Clarification: {analysis.ambiguity.needs_clarification}")
        if analysis.ambiguity.questions:
            print(f"Question: {analysis.ambiguity.questions[0].question}")

        eval_entry = {
            "id": item["id"],
            "description": item["description"],
            "prompt": p,
            "category": analysis.classification.category.value,
            "deliverable_type": deliv.deliverable_type if deliv else None,
            "intended_outcome": deliv.intended_outcome if deliv else None,
            "is_functionally_complete": deliv.is_functionally_complete if deliv else None,
            "ambiguity_score": analysis.ambiguity.score,
            "needs_clarification": analysis.ambiguity.needs_clarification,
            "grounded": analysis.grounding.is_grounded,
            "clarification_question": analysis.ambiguity.questions[0].question if analysis.ambiguity.questions else None,
            "quality_score": None,
            "enhanced_prompt": "",
        }

        # If clarification is NOT needed, run enhancement, validation, and scoring
        if not analysis.ambiguity.needs_clarification:
            _, enhanced = await enhancer.enhance(p, existing_analysis=analysis)
            val = await validator.validate(p, analysis, enhanced)
            print(f"Validation Is Valid: {val.is_valid}")
            eval_entry["validation_is_valid"] = val.is_valid
            eval_entry["enhanced_prompt"] = enhanced

            if val.is_valid:
                report = await scorer.score(p, analysis, enhanced, val)
                print(f"Quality Score: {report.overall_score}")
                print(f"Clarity: {report.clarity.score}, Specificity: {report.specificity.score}, Output Def: {report.output_definition.score}")
                eval_entry["quality_score"] = report.overall_score
                eval_entry["dimensions"] = {
                    "clarity": report.clarity.score,
                    "specificity": report.specificity.score,
                    "output_definition": report.output_definition.score,
                    "intent_preservation": report.intent_preservation.score,
                    "requirement_coverage": report.requirement_coverage.score,
                }

        results.append(eval_entry)

    # Stability run on complete request vs incomplete request with reasonable assumptions
    print("\n==========================================")
    print("STABILITY RUN ON COMPLETE PROMPT")
    print("==========================================")
    complete_p = "compute the factorial of 10"
    runs = []
    spec_scores = []
    out_scores = []
    for r in range(2):
        a = await analyzer.analyze(complete_p)
        _, enh = await enhancer.enhance(complete_p, existing_analysis=a)
        v = await validator.validate(complete_p, a, enh)
        rep = await scorer.score(complete_p, a, enh, v)
        runs.append(rep.overall_score)
        spec_scores.append(rep.specificity.score)
        out_scores.append(rep.output_definition.score)
        print(f"Run {r+1}: Overall={rep.overall_score}, Specificity={rep.specificity.score}, OutputDef={rep.output_definition.score}")

    output_data = {
        "prompts": results,
        "stability": {
            "prompt": complete_p,
            "runs": runs,
            "variance": max(runs) - min(runs),
            "avg_specificity": sum(spec_scores) / len(spec_scores),
            "avg_output_definition": sum(out_scores) / len(out_scores),
        }
    }

    with open("scratch/functional_live_eval_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\nSaved live eval results to scratch/functional_live_eval_results.json")

if __name__ == "__main__":
    asyncio.run(run_live_eval())
