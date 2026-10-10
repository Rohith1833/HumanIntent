import logging
from typing import List, Optional, Tuple
from dataset_factory.config import DatasetFactoryConfig
from dataset_factory.curation import curate_candidate
from dataset_factory.schemas import (
    CurationStatus,
    DatasetRecord,
    RejectionReason,
    SeedIntent,
    SourceType,
)

logger = logging.getLogger("dataset_factory.filters")


def check_intent_preservation(
    original_prompt: str,
    seed: SeedIntent,
) -> bool:
    """Validate that the synthetic prompt preserves the core semantic intent of the seed."""
    prompt_lower = original_prompt.lower()

    # If seed has core requirements, at least one core requirement or related keyword should be reflected
    if seed.core_requirements:
        found_any = False
        for req in seed.core_requirements:
            req_words = [w.lower() for w in req.split() if len(w) > 3]
            if not req_words:
                found_any = True
                break
            if any(w in prompt_lower for w in req_words):
                found_any = True
                break
        if not found_any:
            # If prompt doesn't mention any core requirement words, check intent words
            intent_words = [w.lower() for w in seed.intent.split() if len(w) > 3]
            if not any(w in prompt_lower for w in intent_words):
                return False

    return True


def filter_candidate(
    record: DatasetRecord,
    config: DatasetFactoryConfig,
    seed: Optional[SeedIntent] = None,
) -> Tuple[CurationStatus, List[RejectionReason]]:
    """
    Execute staged quality, safety, and invariant curation filters on a dataset candidate.
    Adheres strictly to the Curation Pipeline Order:
    Schema -> Consent -> Privacy -> Seed Check -> Teacher Evaluation -> Dataset Curation Gate -> Quality Threshold.
    """
    rejection_reasons: List[RejectionReason] = []

    # 1. Empty / Malformed Check
    prompt = record.original_prompt.strip()
    if not prompt or len(prompt) < 4:
        rejection_reasons.append(RejectionReason.MALFORMED)
        return CurationStatus.REJECTED, rejection_reasons

    # 2. Consent Check for Real Data
    if record.source_type == SourceType.CONSENTED_REAL:
        if not record.source_id:
            rejection_reasons.append(RejectionReason.NO_CONSENT)
            return CurationStatus.REJECTED, rejection_reasons

    # 3. Privacy Sanitation Check
    # High-risk secrets are intercepted during sanitization; verify no obvious secrets remained
    if "-----BEGIN" in prompt:
        rejection_reasons.append(RejectionReason.PRIVACY_RISK)
        return CurationStatus.REJECTED, rejection_reasons

    # 4. Intent Preservation Check (Synthetic)
    if seed and record.source_type == SourceType.SYNTHETIC:
        if not check_intent_preservation(prompt, seed):
            rejection_reasons.append(RejectionReason.INTENT_DRIFT)
            return CurationStatus.REJECTED, rejection_reasons

    # 5. Clarification Branch
    if record.clarification_required:
        curation_report = curate_candidate(
            original_prompt=record.original_prompt,
            target_prompt=None,
            category=record.category,
            normalized_intent=record.normalized_intent,
            seed=seed,
            clarification_required=True,
            clarification_questions=record.clarification_questions,
            already_good=record.labels.already_good,
        )
        record.curation_report = curation_report
        if not curation_report.is_acceptable:
            for issue in curation_report.issues:
                if issue.severity == "ERROR":
                    try:
                        rejection_reasons.append(RejectionReason(issue.type.value))
                    except ValueError:
                        rejection_reasons.append(RejectionReason.ANALYSIS_FAILED)
            if not rejection_reasons:
                rejection_reasons.append(RejectionReason.ANALYSIS_FAILED)
            return CurationStatus.REJECTED, rejection_reasons
        return CurationStatus.ACCEPTED, []

    # 6. Teacher Evaluation Check (for positive SFT)
    if not record.quality.validation_passed:
        if not record.normalized_intent:
            rejection_reasons.append(RejectionReason.ANALYSIS_FAILED)
        else:
            rejection_reasons.append(RejectionReason.VALIDATION_FAILED)
        return CurationStatus.REJECTED, rejection_reasons

    # 7. Offline Independent Dataset Curation Gate (Positive Target)
    # DO NOT TRUST validation success or quality score >= 75 as sufficient evidence!
    # Every positive enhancement target must independently pass curation.
    curation_report = curate_candidate(
        original_prompt=record.original_prompt,
        target_prompt=record.enhanced_prompt,
        category=record.category,
        normalized_intent=record.normalized_intent,
        seed=seed,
        clarification_required=False,
        clarification_questions=None,
        already_good=record.labels.already_good,
    )
    record.curation_report = curation_report

    if not curation_report.is_acceptable:
        for issue in curation_report.issues:
            if issue.severity == "ERROR":
                try:
                    rejection_reasons.append(RejectionReason(issue.type.value))
                except ValueError:
                    rejection_reasons.append(RejectionReason.LOW_QUALITY)
        if not rejection_reasons:
            rejection_reasons.append(RejectionReason.LOW_QUALITY)
        return CurationStatus.REJECTED, rejection_reasons

    # 8. Enhanced Prompt Presence
    if not record.enhanced_prompt or not record.enhanced_prompt.strip():
        rejection_reasons.append(RejectionReason.MALFORMED)
        return CurationStatus.REJECTED, rejection_reasons

    # 9. Quality Threshold Gate
    if record.quality.quality_score < config.minimum_quality_score:
        rejection_reasons.append(RejectionReason.LOW_QUALITY)
        return CurationStatus.REJECTED, rejection_reasons

    # 10. Prompt Inflation Guard
    if record.quality.inflation_ratio is not None:
        if record.quality.inflation_ratio > config.max_inflation_ratio:
            # Short inputs naturally have higher inflation; allow if input is very brief (< 40 chars)
            if len(prompt) >= 40:
                rejection_reasons.append(RejectionReason.PROMPT_INFLATION)
                return CurationStatus.REJECTED, rejection_reasons

    return CurationStatus.ACCEPTED, []
