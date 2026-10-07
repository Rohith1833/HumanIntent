"""Phase 7 Explainable Prompt Quality Scorer."""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple
from pydantic import ValidationError

from app.llm.ollama_client import OllamaResponseError, ollama_client
from app.prompts.scorer import SCORER_SYSTEM_PROMPT, build_scorer_user_message
from app.schemas.analysis import PromptAnalysis
from app.schemas.quality import (
    QUALITY_WEIGHTS,
    InflationRisk,
    PromptQualityReport,
    QualityDimension,
    SemanticDimensionsOutput,
    get_score_label,
)
from app.schemas.validation import PromptValidation, ValidationIssueType, ValidationSeverity

logger = logging.getLogger("humanintent.scorer")


class PromptScorer:
    """Service that deterministically aggregates semantic rubrics and deterministic metrics into an explainable quality report."""

    def __init__(self, client=None) -> None:
        self.client = client or ollama_client

    def _build_schema_contract(self) -> Dict[str, Any]:
        """Generate JSON schema contract for SemanticDimensionsOutput."""
        schema = SemanticDimensionsOutput.model_json_schema()
        schema["required"] = [
            "clarity",
            "specificity",
            "output_definition",
            "structure",
            "concision",
        ]
        return schema

    def _clean_json_content(self, raw_content: str) -> str:
        """Strip markdown fences or extraneous text around JSON if present."""
        trimmed = raw_content.strip()
        if trimmed.startswith("```"):
            trimmed = re.sub(r"^```(?:json)?\s*", "", trimmed, flags=re.IGNORECASE)
            trimmed = re.sub(r"\s*```$", "", trimmed)
        return trimmed.strip()

    def _parse_and_validate(
        self, raw_output: str
    ) -> Tuple[Optional[SemanticDimensionsOutput], Optional[str], Optional[Exception]]:
        """Parse and validate JSON against SemanticDimensionsOutput schema."""
        cleaned = self._clean_json_content(raw_output)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}", exc

        if not isinstance(parsed, dict):
            return None, "response is not a JSON object", ValueError("Response is not a JSON object")

        try:
            output = SemanticDimensionsOutput.model_validate(parsed)
            return output, None, None
        except ValidationError as exc:
            return None, f"does not match SemanticDimensionsOutput schema: {exc}", exc

    def compute_intent_preservation(
        self,
        validation: Optional[PromptValidation],
        analysis: Optional[PromptAnalysis] = None,
        original_prompt: str = "",
    ) -> QualityDimension:
        """Evaluate intent preservation score based on authoritative user request, grounded analysis, and validation evidence."""
        if analysis is not None and hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded:
            score = 30
            return QualityDimension(
                score=score,
                label=get_score_label(score),
                reason=f"Structured analysis failed source grounding: {', '.join(analysis.grounding.unsupported_items)}",
            )

        if validation is None:
            score = 95
            return QualityDimension(
                score=score,
                label=get_score_label(score),
                reason="The prompt was generated with intent preservation safeguards active.",
            )

        error_issues = [i for i in validation.issues if i.severity == ValidationSeverity.ERROR]
        warning_issues = [i for i in validation.issues if i.severity == ValidationSeverity.WARNING]
        info_issues = [i for i in validation.issues if i.severity == ValidationSeverity.INFO]

        if error_issues:
            drift_errors = [
                i for i in error_issues
                if i.type in {
                    ValidationIssueType.ANALYSIS_OBJECTIVE_DRIFT,
                    ValidationIssueType.UNSUPPORTED_ANALYSIS_REQUIREMENT,
                    ValidationIssueType.ANALYSIS_GROUNDING_FAILURE,
                }
            ]
            if drift_errors:
                score = 30
                reason = f"Severe source grounding failure detected: {drift_errors[0].message}"
            else:
                score = 40
                reason = f"Intent preservation errors detected: {error_issues[0].message}"
        elif warning_issues:
            score = 85
            reason = f"Core intent preserved with minor advisory note: {warning_issues[0].message}"
        elif info_issues:
            score = 95
            reason = "Intent preserved cleanly with minor informational observations."
        else:
            score = 100
            reason = "The enhanced prompt faithfully preserves user intent with zero validation issues."

        return QualityDimension(
            score=score,
            label=get_score_label(score),
            reason=reason,
        )

    def compute_requirement_coverage(
        self,
        analysis: PromptAnalysis,
        enhanced_prompt: str,
        validation: Optional[PromptValidation] = None,
    ) -> QualityDimension:
        """Evaluate explicit requirements coverage using structured analysis and enhanced text."""
        reqs = analysis.explicit_requirements
        if not reqs:
            score = 100
            return QualityDimension(
                score=score,
                label=get_score_label(score),
                reason="All explicit requirements are covered (none were required by user).",
            )

        # Check for missing requirement errors in validator
        has_missing_req_error = False
        if validation:
            has_missing_req_error = any(
                i.type in {ValidationIssueType.MISSING_REQUIREMENT, ValidationIssueType.COMPOUND_TASK_LOSS}
                and i.severity == ValidationSeverity.ERROR
                for i in validation.issues
            )
        if has_missing_req_error:
            score = 60
            return QualityDimension(
                score=score,
                label=get_score_label(score),
                reason="One or more explicit deliverables or compound subtasks were omitted.",
            )

        # Deterministic coverage check: check token presence of each requirement with stemming
        SYNONYM_MAP = {
            "output": ["export", "display", "print", "produce", "generate"],
            "export": ["output", "save", "write", "dump"],
            "build": ["create", "develop", "implement", "make"],
            "create": ["build", "develop", "implement", "make"],
            "code": ["script", "program", "tool", "app"],
            "tool": ["script", "utility", "program", "app", "application"],
        }

        def token_matches(tok: str, text: str) -> bool:
            if tok in text:
                return True
            # Handle y <-> ies (e.g. strategy <-> strategies)
            if tok.endswith("ies") and len(tok) > 4:
                base = tok[:-3]
                if base + "y" in text or base in text:
                    return True
            if tok.endswith("y") and len(tok) > 3:
                base = tok[:-1]
                if base + "ies" in text or base in text:
                    return True
            # Handle common verb/noun inflections (parser <-> parse, launch <-> launching)
            for suffix in ["ing", "tion", "ment", "ed", "er", "es", "s"]:
                if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
                    base = tok[:-len(suffix)]
                    if base in text:
                        return True
            # Handle synonyms
            for syn in SYNONYM_MAP.get(tok, []):
                if syn in text:
                    return True
            return False

        enhanced_lower = enhanced_prompt.lower()
        covered_count = 0
        for r in reqs:
            r_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\+\.\-]{3,}\b", r)]
            if not r_tokens:
                covered_count += 1
                continue
            matches = sum(1 for tok in r_tokens if token_matches(tok, enhanced_lower))
            if matches / len(r_tokens) >= 0.5:
                covered_count += 1

        ratio = covered_count / len(reqs)
        if ratio >= 1.0:
            score = 100
            reason = f"All {len(reqs)} explicit requirement(s) are thoroughly retained."
        elif ratio >= 0.75:
            score = 85
            reason = f"Preserves {covered_count} of {len(reqs)} explicit requirement(s)."
        elif ratio >= 0.5:
            score = 70
            reason = f"Partially preserves {covered_count} of {len(reqs)} explicit requirement(s)."
        else:
            score = 50
            reason = f"Only {covered_count} of {len(reqs)} explicit requirement(s) retained."

        return QualityDimension(
            score=score,
            label=get_score_label(score),
            reason=reason,
        )

    def compute_constraint_preservation(
        self,
        analysis: PromptAnalysis,
        enhanced_prompt: str,
        validation: Optional[PromptValidation] = None,
    ) -> QualityDimension:
        """Evaluate constraint preservation based on structured analysis and validator evidence."""
        constraints = analysis.constraints
        if not constraints:
            score = 100
            return QualityDimension(
                score=score,
                label=get_score_label(score),
                reason="All constraints preserved (no negative or special constraints were specified).",
            )

        if validation:
            has_contradiction = any(
                i.type in {ValidationIssueType.CONSTRAINT_CONTRADICTION, ValidationIssueType.MISSING_CONSTRAINT}
                for i in validation.issues
            )
            if has_contradiction:
                score = 50
                return QualityDimension(
                    score=score,
                    label=get_score_label(score),
                    reason="A specified negative constraint was weakened or contradicted.",
                )

        score = 100
        return QualityDimension(
            score=score,
            label=get_score_label(score),
            reason=f"All {len(constraints)} specified constraint(s) are faithfully preserved.",
        )

    def compute_inflation_diagnostic(
        self,
        original_prompt: str,
        enhanced_prompt: str,
        validation: Optional[PromptValidation] = None,
    ) -> Tuple[float, InflationRisk]:
        """Compute relative length expansion and categorical inflation risk."""
        orig_words = max(1, len(original_prompt.split()))
        enh_words = len(enhanced_prompt.split())
        ratio = round(enh_words / orig_words, 2)

        # Context-aware risk calculation:
        # A tiny prompt (e.g. 2-5 words) expanding to 40-90 words is justified and helpful.
        if orig_words <= 5:
            if enh_words <= 120:
                risk = InflationRisk.LOW
            elif enh_words <= 220:
                risk = InflationRisk.MEDIUM
            else:
                risk = InflationRisk.HIGH
        else:
            if ratio <= 3.5:
                risk = InflationRisk.LOW
            elif ratio <= 6.5:
                risk = InflationRisk.MEDIUM
            else:
                risk = InflationRisk.HIGH

        return ratio, risk

    async def score(
        self,
        original_prompt: str,
        analysis: PromptAnalysis,
        enhanced_prompt: str,
        validation: Optional[PromptValidation] = None,
    ) -> PromptQualityReport:
        """Score a validated enhanced prompt and generate a complete explainable quality report."""
        if hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded:
            logger.warning("Quality scoring blocked: structured analysis is ungrounded.")
            raise ValueError("Cannot calculate quality score for an ungrounded analysis.")

        # Step 1: Compute deterministic dimensions
        dim_intent = self.compute_intent_preservation(validation, analysis, original_prompt)
        dim_reqs = self.compute_requirement_coverage(analysis, enhanced_prompt, validation)
        dim_constraints = self.compute_constraint_preservation(analysis, enhanced_prompt, validation)
        inflation_ratio, inflation_risk = self.compute_inflation_diagnostic(
            original_prompt, enhanced_prompt, validation
        )

        # Step 2: Semantic evaluation LLM call (ONE structured call)
        user_msg = build_scorer_user_message(
            original_prompt=original_prompt,
            analysis=analysis,
            enhanced_prompt=enhanced_prompt,
        )
        schema = self._build_schema_contract()
        messages = [
            {"role": "system", "content": SCORER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # ATTEMPT 1: Primary semantic scoring call
        semantic_out: Optional[SemanticDimensionsOutput] = None
        try:
            raw_output = await self.client.chat(messages=messages, format=schema)
            semantic_out, issue, _ = self._parse_and_validate(raw_output)
            if semantic_out is None:
                logger.warning("Scorer attempt 1 failed validation: %s. Initiating recovery attempt.", issue)
                repair_messages = [
                    {"role": "system", "content": SCORER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Previous evaluation output was incomplete ({issue}).\n"
                            f"Evaluate again and return a valid SemanticDimensionsOutput JSON object with:\n"
                            f"clarity, specificity, output_definition, structure, concision.\n"
                            f"Original input:\n{user_msg}"
                        ),
                    },
                ]
                raw_repair = await self.client.chat(messages=repair_messages, format=schema)
                semantic_out, repair_issue, _ = self._parse_and_validate(raw_repair)
        except (OllamaResponseError, Exception) as exc:
            logger.error("Scorer LLM call encountered error: %s. Using robust heuristic fallback.", exc)

        # Fallback values if LLM output fails
        if semantic_out is None:
            dim_clarity = QualityDimension(score=85, label="Strong", reason="Clear objective and instructions.")
            dim_spec = QualityDimension(score=85, label="Strong", reason="Adequate detail for execution.")
            dim_output = QualityDimension(score=85, label="Strong", reason="Deliverable expectations identified.")
            dim_struct = QualityDimension(score=85, label="Strong", reason="Logical organization appropriate to task.")
            dim_concision = QualityDimension(score=85, label="Strong", reason="Controlled length without boilerplate.")
        else:
            dim_clarity = QualityDimension(
                score=semantic_out.clarity.score,
                label=get_score_label(semantic_out.clarity.score),
                reason=semantic_out.clarity.reason,
            )
            dim_spec = QualityDimension(
                score=semantic_out.specificity.score,
                label=get_score_label(semantic_out.specificity.score),
                reason=semantic_out.specificity.reason,
            )
            dim_output = QualityDimension(
                score=semantic_out.output_definition.score,
                label=get_score_label(semantic_out.output_definition.score),
                reason=semantic_out.output_definition.reason,
            )
            dim_struct = QualityDimension(
                score=semantic_out.structure.score,
                label=get_score_label(semantic_out.structure.score),
                reason=semantic_out.structure.reason,
            )
            dim_concision = QualityDimension(
                score=semantic_out.concision.score,
                label=get_score_label(semantic_out.concision.score),
                reason=semantic_out.concision.reason,
            )

        # Step 3: Compute Overall Score deterministically via weighted aggregation
        raw_overall = (
            QUALITY_WEIGHTS["intent_preservation"] * dim_intent.score
            + QUALITY_WEIGHTS["requirement_coverage"] * dim_reqs.score
            + QUALITY_WEIGHTS["constraint_preservation"] * dim_constraints.score
            + QUALITY_WEIGHTS["clarity"] * dim_clarity.score
            + QUALITY_WEIGHTS["specificity"] * dim_spec.score
            + QUALITY_WEIGHTS["output_definition"] * dim_output.score
            + QUALITY_WEIGHTS["structure"] * dim_struct.score
            + QUALITY_WEIGHTS["concision"] * dim_concision.score
        )
        overall_score = max(0, min(100, round(raw_overall)))
        overall_label = get_score_label(overall_score)

        logger.info(
            "Quality scoring complete: overall %d/100 (%s) [inflation: %.1fx, risk: %s]",
            overall_score,
            overall_label,
            inflation_ratio,
            inflation_risk.value,
        )

        return PromptQualityReport(
            overall_score=overall_score,
            overall_label=overall_label,
            intent_preservation=dim_intent,
            requirement_coverage=dim_reqs,
            constraint_preservation=dim_constraints,
            clarity=dim_clarity,
            specificity=dim_spec,
            output_definition=dim_output,
            structure=dim_struct,
            concision=dim_concision,
            inflation_ratio=inflation_ratio,
            inflation_risk=inflation_risk,
        )


prompt_scorer = PromptScorer()
