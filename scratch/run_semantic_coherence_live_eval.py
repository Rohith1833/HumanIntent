import asyncio
import json
import logging
import os
import sys

os.environ["OLLAMA_MODEL"] = "llama3"

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.analyzer import prompt_analyzer
from app.core.enhancer import prompt_enhancer
from app.core.validator import prompt_validator
from app.core.scorer import prompt_scorer
from app.templates.selector import get_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_eval")

PROMPTS = [
    {
        "category": "CODING",
        "prompt": "Implement a thread-safe token bucket rate limiter in Python with acquire and refill methods.",
    },
    {
        "category": "DEBUGGING",
        "prompt": "Why does this Python multiprocessing worker freeze when using a shared queue on Windows?",
    },
    {
        "category": "WRITING",
        "prompt": "Draft a formal announcement letter welcoming our new vice president of engineering to the executive team.",
    },
    {
        "category": "EDUCATION",
        "prompt": "Explain how zero-knowledge proofs work using the classic cave analogy for high school students.",
    },
    {
        "category": "RESEARCH",
        "prompt": "Investigate and compare the empirical throughput tradeoffs between B-tree and LSM-tree storage engines under write-heavy workloads.",
    },
    {
        "category": "PLANNING",
        "prompt": "Develop a phased 30-day onboarding roadmap for a junior backend engineer joining an agile team.",
    },
    {
        "category": "DATA_ANALYSIS",
        "prompt": "Calculate the median latency, p99, and standard deviation from these response times: 12ms, 15ms, 14ms, 120ms, 18ms.",
    },
    {
        "category": "CREATIVE_IDEATION",
        "prompt": "Brainstorm ten innovative theme concepts and gameplay mechanics for a cooperative deck-building tabletop game.",
    },
    {
        "category": "IMAGE_GENERATION",
        "prompt": "Create an image prompt for a cozy bookstore cafe during a rainy afternoon with warm interior lighting.",
    },
    {
        "category": "SUMMARIZATION",
        "prompt": "Summarize the key findings from this abstract: Recent advances in solar cell efficiency have demonstrated a 28% tandem perovskite-silicon yield under laboratory illumination, though thermal stability remains limited beyond 500 hours.",
    },
    {
        "category": "GENERAL",
        "prompt": "What factors should an organization weigh when transitioning from centralized to decentralized operations?",
    },
]

async def run_eval():
    enhancer = prompt_enhancer
    enhancer.validator = prompt_validator
    enhancer.scorer = prompt_scorer

    results = []

    for item in PROMPTS:
        exp_cat = item["category"]
        text = item["prompt"]
        print(f"\n=======================================================")
        print(f"RUNNING LIVE EVAL FOR: {exp_cat}")
        print(f"PROMPT: {text}")
        print(f"=======================================================")

            res = await enhancer.enhance(
                user_prompt=text,
                allow_reasonable_assumptions=True,
                validate=True,
            )
            analysis = res.analysis
            template_obj = get_template(analysis.classification.category) if analysis.classification else None
            validation = res.validation
            raw_category = getattr(analysis, "raw_category", "UNKNOWN") or "UNKNOWN"

            val_status = "VALID" if (validation and validation.is_valid) else "INVALID"

            record = {
                "input": text,
                "expected_category": exp_cat,
                "primary_action": analysis.intent_anchor.primary_action if analysis.intent_anchor else "None",
                "intent_anchor": analysis.intent_anchor.model_dump() if analysis.intent_anchor else None,
                "raw_model_category": raw_category,
                "final_category": analysis.classification.category.value if analysis.classification else "None",
                "confidence": analysis.classification.confidence if analysis.classification else 0.0,
                "objective": analysis.objective,
                "explicit_requirements": analysis.explicit_requirements,
                "missing_context": analysis.missing_context,
                "functional_completeness": analysis.deliverable_semantics.is_functionally_complete if analysis.deliverable_semantics else False,
                "ambiguity_score": analysis.ambiguity.score,
                "ambiguity_level": analysis.ambiguity.level.value,
                "needs_clarification": analysis.ambiguity.needs_clarification,
                "template": template_obj.name if template_obj else "None",
                "validation_status": val_status,
                "enhanced_prompt_sample": res.enhanced_prompt[:200] if res.enhanced_prompt else "",
            }
            results.append(record)
            print(f"RAW MODEL CATEGORY: {raw_category}")
            print(f"FINAL CATEGORY: {record['final_category']}")
            print(f"ACTION ANCHOR: {record['primary_action']}")
            print(f"VALIDATION STATUS: {record['validation_status']}")
            print(f"OBJECTIVE: {record['objective']}")
            print(f"MISSING CONTEXT: {record['missing_context']}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({
                "input": text,
                "expected_category": exp_cat,
                "error": str(exc),
            })

    output_path = os.path.join(os.path.dirname(__file__), "live_eval_coherence_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nCompleted live evaluation across all 11 categories. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_eval())
