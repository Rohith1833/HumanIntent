"""
Stochastic Stability & Repeatability Evaluation Module for Phase 10.

Evaluates category stability, clarification stability, and quality score variance
across repeated runs on a subset of benchmark cases.
"""

import asyncio
import logging
from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field

from evaluation.schemas import BenchmarkCase
from app.core.analyzer import PromptAnalyzer
from app.core.scorer import PromptScorer
from app.core.enhancer import PromptEnhancer

logger = logging.getLogger(__name__)


class StabilityMetrics(BaseModel):
    """Metrics quantifying stochastic stability across repeated runs."""
    total_cases_evaluated: int = 0
    repetitions_per_case: int = 3
    category_stability_rate: float = Field(
        ..., description="Proportion of cases where category remained identical across all runs"
    )
    clarification_stability_rate: float = Field(
        ..., description="Proportion of cases where clarification decision remained identical across all runs"
    )
    mean_quality_score_std: float = Field(
        ..., description="Mean standard deviation of quality scores across runs"
    )
    max_quality_score_std: float = Field(
        ..., description="Max standard deviation of quality scores across runs"
    )
    case_stability_details: List[Dict[str, Any]] = Field(default_factory=list)


async def run_stability_evaluation(
    cases: List[BenchmarkCase],
    repetitions: int = 3,
    model: str = "llama3:latest",
) -> StabilityMetrics:
    """
    Evaluates stability by executing each case multiple times and measuring variance.
    """
    analyzer = PromptAnalyzer()
    scorer = PromptScorer()

    case_details = []
    category_identical_count = 0
    clarification_identical_count = 0
    quality_stds = []

    for idx, case in enumerate(cases, 1):
        categories_seen = []
        clarifications_seen = []
        quality_scores = []

        logger.info(
            "Evaluating stability for case %d/%d (%s) over %d runs...",
            idx, len(cases), case.case_id, repetitions,
        )

        for rep in range(repetitions):
            # Run analysis
            analysis = await analyzer.analyze(case.user_prompt)
            categories_seen.append(analysis.classification.category.value)

            # Check clarification gate
            needs_clar = bool(analysis.ambiguity.needs_clarification)
            clarifications_seen.append(needs_clar)

            # Quality scoring on structured analysis
            dim_reqs = scorer.compute_requirement_coverage(analysis, case.user_prompt, None)
            dim_con = scorer.compute_constraint_preservation(analysis, case.user_prompt, None)
            q_val = float(dim_reqs.score + dim_con.score) / 2.0
            quality_scores.append(q_val)

        cat_stable = len(set(categories_seen)) == 1
        clar_stable = len(set(clarifications_seen)) == 1
        q_std = float(np.std(quality_scores)) if len(quality_scores) > 1 else 0.0

        if cat_stable:
            category_identical_count += 1
        if clar_stable:
            clarification_identical_count += 1
        quality_stds.append(q_std)

        case_details.append({
            "case_id": case.case_id,
            "category_stable": cat_stable,
            "categories": categories_seen,
            "clarification_stable": clar_stable,
            "clarification_decisions": clarifications_seen,
            "quality_scores": quality_scores,
            "quality_std": round(q_std, 4),
        })

    n = len(cases)
    cat_rate = (category_identical_count / n) if n > 0 else 1.0
    clar_rate = (clarification_identical_count / n) if n > 0 else 1.0
    mean_std = float(np.mean(quality_stds)) if quality_stds else 0.0
    max_std = float(np.max(quality_stds)) if quality_stds else 0.0

    return StabilityMetrics(
        total_cases_evaluated=n,
        repetitions_per_case=repetitions,
        category_stability_rate=cat_rate,
        clarification_stability_rate=clar_rate,
        mean_quality_score_std=round(mean_std, 4),
        max_quality_score_std=round(max_std, 4),
        case_stability_details=case_details,
    )
