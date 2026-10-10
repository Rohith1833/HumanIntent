"""
Independent Semantic Judge & Blinded Pairwise Evaluation for Phase 10.
Does not trust HumanIntent's internal scorer; uses an external structured judge prompt
with A/B blinding and position-bias checking.
"""

import json
import logging
import random
import re
from typing import Optional, Tuple

from app.llm.ollama_client import OllamaClient, ollama_client
from evaluation.schemas import (
    BenchmarkCase,
    CasePrediction,
    PairwiseJudgeResult,
    PairwisePreference,
    SemanticJudgeResult,
)

logger = logging.getLogger("evaluation.judges")

SINGLE_JUDGE_SYSTEM = (
    "You are an impartial, highly rigorous AI evaluation judge. Your task is to evaluate "
    "whether an enhanced prompt faithfully preserves the user's authentic intent, satisfies all "
    "constraints, and avoids inventing unrequested features. "
    "You must return ONLY a valid JSON object matching the requested schema without any markdown commentary."
)

PAIRWISE_JUDGE_SYSTEM = (
    "You are an impartial AI evaluation judge comparing two alternative prompt formulations. "
    "Your goal is to decide which prompt is superior for an AI assistant to faithfully execute "
    "the user's authentic intent. The superior prompt must:\n"
    "1. Preserve all user requirements and constraints\n"
    "2. Not invent unrequested requirements or features\n"
    "3. Be clear, structured, and free of useless bloat\n"
    "Return ONLY a valid JSON object: {\"winner\": \"Candidate 1\" | \"Candidate 2\" | \"TIE\", \"reason\": \"<brief reason>\"}."
)


def _clean_json_str(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


async def evaluate_single_semantic(
    case: BenchmarkCase,
    prediction: CasePrediction,
    client: Optional[OllamaClient] = None,
    judge_model: str = "llama3:latest",
) -> SemanticJudgeResult:
    """Evaluates semantic fidelity and hallucination of a single prediction."""
    c = client or ollama_client
    if prediction.clarification_requested or not prediction.enhanced_prompt:
        return SemanticJudgeResult(
            intent_preserved=True,
            objective_drift=False,
            unsupported_requirements=[],
            missing_requirements=[],
            constraint_violations=[],
            quantity_violations=[],
            quality_preference="TIE",
            explanation_summary="Clarification requested by system.",
        )

    user_msg = (
        f"Original User Prompt:\n\"\"\"{case.user_prompt}\"\"\"\n\n"
        f"Enhanced Candidate Prompt:\n\"\"\"{prediction.enhanced_prompt}\"\"\"\n\n"
        f"Expected Requirements to Preserve:\n{case.expected.must_preserve}\n\n"
        f"Unrequested Features to Avoid:\n{case.expected.must_not_invent}\n\n"
        f"Constraints:\n{case.expected.constraints}\n\n"
        "Return a JSON object with fields:\n"
        "{\n"
        '  "intent_preserved": true/false,\n'
        '  "objective_drift": true/false,\n'
        '  "unsupported_requirements": ["list", "of", "hallucinations"],\n'
        '  "missing_requirements": ["list", "of", "dropped", "items"],\n'
        '  "constraint_violations": ["list"],\n'
        '  "quantity_violations": ["list"],\n'
        '  "quality_preference": "RAW" | "ENHANCED" | "TIE",\n'
        '  "explanation_summary": "one brief sentence"\n'
        "}"
    )

    try:
        raw = await c.chat(
            messages=[
                {"role": "system", "content": SINGLE_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            model=judge_model,
        )
        data = json.loads(_clean_json_str(raw))
        return SemanticJudgeResult.model_validate(data)
    except Exception as e:
        logger.debug("Semantic judge fallback error: %s", e)
        # Deterministic invariant check fallback
        unsupported = [m for m in case.expected.must_not_invent if m.lower() in prediction.enhanced_prompt.lower()]
        missing = [r for r in case.expected.must_preserve if not any(w in prediction.enhanced_prompt.lower() for w in r.lower().split() if len(w) > 3)]
        return SemanticJudgeResult(
            intent_preserved=len(missing) == 0,
            objective_drift=False,
            unsupported_requirements=unsupported,
            missing_requirements=missing,
            constraint_violations=[],
            quantity_violations=[],
            quality_preference="ENHANCED" if len(missing) == 0 and len(unsupported) == 0 else "TIE",
            explanation_summary=f"Automated invariant evaluation: {len(missing)} missing, {len(unsupported)} unsupported.",
        )


async def evaluate_pairwise_blinded(
    case: BenchmarkCase,
    candidate_a: CasePrediction,
    candidate_b: CasePrediction,
    client: Optional[OllamaClient] = None,
    judge_model: str = "llama3:latest",
    random_seed: int = 42,
    swap_override: Optional[bool] = None,
) -> PairwiseJudgeResult:
    """
    Blinded pairwise judge comparison between candidate_a and candidate_b.
    Randomly assigns candidates to 'Candidate 1' and 'Candidate 2' to eliminate name bias.
    Supports swap_override to test position bias across forward and swapped presentations.
    """
    c = client or ollama_client
    rng = random.Random(random_seed + hash(case.case_id))

    # 50/50 swap assignment or explicit override
    swap = swap_override if swap_override is not None else rng.choice([True, False])
    if swap:
        first_pred, second_pred = candidate_b, candidate_a
        first_baseline, second_baseline = candidate_b.baseline.value, candidate_a.baseline.value
    else:
        first_pred, second_pred = candidate_a, candidate_b
        first_baseline, second_baseline = candidate_a.baseline.value, candidate_b.baseline.value

    user_msg = (
        f"Original User Prompt:\n\"\"\"{case.user_prompt}\"\"\"\n\n"
        f"Candidate 1:\n\"\"\"{first_pred.enhanced_prompt}\"\"\"\n\n"
        f"Candidate 2:\n\"\"\"{second_pred.enhanced_prompt}\"\"\"\n\n"
        "Evaluate which candidate is better. Return ONLY JSON:\n"
        '{"winner": "Candidate 1" | "Candidate 2" | "TIE", "reason": "<brief justification>"}'
    )

    try:
        raw = await c.chat(
            messages=[
                {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            model=judge_model,
        )
        parsed = json.loads(_clean_json_str(raw))
        winner_raw = parsed.get("winner", "TIE").strip()
        reason = parsed.get("reason", "")
    except Exception as e:
        logger.debug("Pairwise judge LLM error (%s), using objective invariant comparison", e)
        # Objective invariant comparison: measure unsupported additions and requirement survival
        a_unsupp = sum(1 for m in case.expected.must_not_invent if m.lower() in first_pred.enhanced_prompt.lower())
        b_unsupp = sum(1 for m in case.expected.must_not_invent if m.lower() in second_pred.enhanced_prompt.lower())
        if a_unsupp < b_unsupp:
            winner_raw = "Candidate 1"
            reason = f"Candidate 1 avoided {b_unsupp - a_unsupp} unsupported requirement(s)."
        elif b_unsupp < a_unsupp:
            winner_raw = "Candidate 2"
            reason = f"Candidate 2 avoided {a_unsupp - b_unsupp} unsupported requirement(s)."
        else:
            winner_raw = "TIE"
            reason = "Equivalent invariant preservation and absence of hallucinations."

    # Decode winner back to actual system
    if "1" in winner_raw:
        chosen_pref = PairwisePreference.B_WINS if swap else PairwisePreference.A_WINS
        decoded_winner = first_baseline
    elif "2" in winner_raw:
        chosen_pref = PairwisePreference.A_WINS if swap else PairwisePreference.B_WINS
        decoded_winner = second_baseline
    else:
        chosen_pref = PairwisePreference.TIE
        decoded_winner = "tie"

    return PairwiseJudgeResult(
        chosen_label=chosen_pref,
        decoded_winner=decoded_winner,
        reasoning=reason,
        candidate_a_baseline=candidate_a.baseline.value,
        candidate_b_baseline=candidate_b.baseline.value,
    )
