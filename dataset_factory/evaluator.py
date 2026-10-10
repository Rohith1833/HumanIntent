import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("dataset_factory.evaluator")


class TeacherEvaluation(BaseModel):
    """Result of running a candidate prompt through the HumanIntent teacher pipeline."""
    status: str = Field(..., description="enhanced, needs_clarification, analysis_failed, validation_failed")
    analysis_dict: Optional[Dict[str, Any]] = None
    enhanced_prompt: Optional[str] = None
    is_valid: bool = False
    validation_issues: List[str] = Field(default_factory=list)
    quality_score: float = 0.0
    inflation_ratio: Optional[float] = None
    repaired: bool = False
    initial_rejected_prompt: Optional[str] = None
    initial_rejection_reason: Optional[str] = None


class PipelineEvaluator:
    """Evaluates candidate prompts using the locked Phases 1-8 HumanIntent pipeline."""

    def __init__(self, cache_dir: str = "datasets/cache", pipeline_version: str = "v1.0"):
        self.cache_dir = cache_dir
        self.pipeline_version = pipeline_version
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, prompt: str) -> str:
        content = f"{prompt}:{self.pipeline_version}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[TeacherEvaluation]:
        path = os.path.join(self.cache_dir, f"eval_{key}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return TeacherEvaluation.model_validate(json.load(f))
            except Exception as e:
                logger.debug("Cache read error for %s: %s", key, e)
        return None

    def _write_cache(self, key: str, result: TeacherEvaluation) -> None:
        path = os.path.join(self.cache_dir, f"eval_{key}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            logger.debug("Cache write error for %s: %s", key, e)

    async def evaluate_prompt(self, prompt: str, force: bool = False) -> TeacherEvaluation:
        """Run candidate prompt through HumanIntent enhancer, validator, and scorer."""
        key = self._cache_key(prompt)
        if not force:
            cached = self._read_cache(key)
            if cached:
                return cached

        # Import in-process pipeline components
        from app.core.enhancer import prompt_enhancer
        from app.core.validator import prompt_validator
        from app.core.scorer import prompt_scorer

        enhancer = prompt_enhancer
        enhancer.validator = prompt_validator
        enhancer.scorer = prompt_scorer

        try:
            res = await enhancer.enhance(
                user_prompt=prompt,
                allow_reasonable_assumptions=False,
                validate=True,
            )

            analysis = res.analysis
            enhanced_prompt = res.enhanced_prompt
            validation = res.validation
            quality = res.quality

            # Determine status based on pipeline execution outcome
            if analysis is None or (hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded):
                status = "analysis_failed"
            elif analysis and analysis.ambiguity and analysis.ambiguity.needs_clarification:
                status = "needs_clarification"
            elif validation and not validation.is_valid:
                status = "validation_failed"
            elif enhanced_prompt:
                status = "enhanced"
            else:
                status = "analysis_failed"

            analysis_dict = analysis.model_dump() if analysis else None
            is_valid = bool(validation and validation.is_valid)
            val_issues = [iss.message for iss in validation.issues] if validation else []
            quality_score = quality.overall_score if quality else 0.0
            inflation_ratio = quality.inflation_ratio if quality else None

            # Track if repair occurred
            repaired = False
            initial_rejected = None
            initial_reason = None
            if getattr(res, "repair_attempted", False) and getattr(res, "initial_enhanced_prompt", None) and res.initial_enhanced_prompt != res.enhanced_prompt:
                repaired = True
                initial_rejected = res.initial_enhanced_prompt
                if getattr(res, "initial_validation", None) and res.initial_validation.issues:
                    err_msgs = [iss.message for iss in res.initial_validation.issues if getattr(iss, "severity", None) and iss.severity.value == "error"]
                    initial_reason = "; ".join(err_msgs) if err_msgs else "Initial draft required repair"
                else:
                    initial_reason = "Initial draft required repair"

            evaluation = TeacherEvaluation(
                status=status,
                analysis_dict=analysis_dict,
                enhanced_prompt=enhanced_prompt,
                is_valid=is_valid,
                validation_issues=val_issues,
                quality_score=quality_score,
                inflation_ratio=inflation_ratio,
                repaired=repaired,
                initial_rejected_prompt=initial_rejected,
                initial_rejection_reason=initial_reason,
            )

        except Exception as e:
            logger.warning("Teacher evaluation failed for prompt '%s': %s", prompt[:50], e)
            evaluation = TeacherEvaluation(
                status="analysis_failed",
                is_valid=False,
                validation_issues=[f"Pipeline error: {str(e)}"],
            )

        self._write_cache(key, evaluation)
        return evaluation
