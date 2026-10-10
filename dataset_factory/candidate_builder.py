from typing import Optional
from dataset_factory.sanitizer import sanitize_text
from dataset_factory.schemas import (
    CurationStatus,
    DatasetRecord,
    NormalizedIntent,
    PrivacyMetadata,
    ProvenanceMetadata,
    QualityMetadata,
    RecordLabels,
    SeedIntent,
    SourceType,
    SyntheticVariant,
    VariantType,
)
from dataset_factory.evaluator import TeacherEvaluation


def build_synthetic_candidate(
    seed: SeedIntent,
    variant: SyntheticVariant,
    evaluation: TeacherEvaluation,
    dataset_version: str = "humanintent-v0.1.0",
    model: str = "llama3:latest",
) -> DatasetRecord:
    """Transform synthetic variant and teacher evaluation into a canonical DatasetRecord."""
    # Sanitize prompt (defense in depth)
    sanitized_res = sanitize_text(variant.prompt)

    # Extract normalized intent from analysis dict
    normalized_intent = None
    if evaluation.analysis_dict:
        ad = evaluation.analysis_dict
        normalized_intent = NormalizedIntent(
            objective=ad.get("objective", ""),
            explicit_requirements=ad.get("explicit_requirements", []),
            constraints=ad.get("constraints", []),
            preferences=ad.get("preferences", []),
            missing_context=ad.get("missing_context", []),
        )

    # Clarification handling
    clarification_required = evaluation.status == "needs_clarification"
    clarification_questions = []
    if clarification_required and evaluation.analysis_dict:
        ambiguity = evaluation.analysis_dict.get("ambiguity", {})
        for q in ambiguity.get("questions", []):
            if isinstance(q, dict) and "question" in q:
                clarification_questions.append(q["question"])
            elif isinstance(q, str):
                clarification_questions.append(q)

    # Behavioral labels
    labels = RecordLabels(
        already_good=(variant.variant_type == VariantType.ALREADY_GOOD),
        compound_request=(variant.variant_type == VariantType.COMPOUND),
        self_correction=(variant.variant_type == VariantType.SELF_CORRECTING),
        ambiguous=(variant.variant_type == VariantType.AMBIGUOUS or clarification_required),
        negative_constraint=(variant.variant_type == VariantType.CONSTRAINED),
        quantitative_constraint=("quantitative" in [t.lower() for t in seed.tags]),
    )

    quality_meta = QualityMetadata(
        validation_passed=evaluation.is_valid,
        quality_score=evaluation.quality_score,
        inflation_ratio=evaluation.inflation_ratio,
        validation_issues=evaluation.validation_issues,
    )

    provenance = ProvenanceMetadata(
        seed_id=seed.seed_id,
        variant_id=variant.variant_id,
        generation_model=model,
        teacher_pipeline_version="humanintent-v1.0",
        validator_version="v1.0",
        dataset_version=dataset_version,
    )

    privacy = PrivacyMetadata(
        sanitized=sanitized_res.sanitized,
        pii_types_removed=sanitized_res.pii_types_removed,
    )

    return DatasetRecord(
        dataset_version=dataset_version,
        source_type=SourceType.SYNTHETIC,
        category=seed.category,
        original_prompt=sanitized_res.text,
        normalized_intent=normalized_intent,
        enhanced_prompt=evaluation.enhanced_prompt,
        clarification_required=clarification_required,
        clarification_questions=clarification_questions,
        quality=quality_meta,
        provenance=provenance,
        privacy=privacy,
        labels=labels,
        curation_status=CurationStatus.RAW,
    )
