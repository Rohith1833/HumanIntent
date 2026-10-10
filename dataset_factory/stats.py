import math
from typing import Any, Dict, List
from dataset_factory.schemas import CurationStatus, DatasetRecord


def compute_dataset_statistics(
    all_records: List[DatasetRecord],
) -> Dict[str, Any]:
    """Compute comprehensive dataset metrics and diversity health checks."""
    total = len(all_records)
    if total == 0:
        return {
            "total_candidates": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "acceptance_rate": 0.0,
            "warnings": [],
        }

    accepted = [r for r in all_records if r.curation_status == CurationStatus.ACCEPTED]
    rejected = [r for r in all_records if r.curation_status == CurationStatus.REJECTED]

    # Category counts
    category_counts: Dict[str, int] = {}
    for r in accepted:
        category_counts[r.category] = category_counts.get(r.category, 0) + 1

    # Source counts
    source_counts: Dict[str, int] = {}
    for r in accepted:
        st = r.source_type.value
        source_counts[st] = source_counts.get(st, 0) + 1

    # Rejection reasons
    rejection_counts: Dict[str, int] = {}
    for r in rejected:
        for rj in r.rejection_reasons:
            rejection_counts[rj.value] = rejection_counts.get(rj.value, 0) + 1

    # Quality distribution
    quality_scores = [r.quality.quality_score for r in accepted if not r.clarification_required]
    mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    min_quality = min(quality_scores) if quality_scores else 0.0
    max_quality = max(quality_scores) if quality_scores else 0.0

    # Expansion / Inflation metrics
    expansion_ratios: List[float] = []
    for r in accepted:
        if r.enhanced_prompt and len(r.original_prompt) > 0:
            expansion = len(r.enhanced_prompt) / len(r.original_prompt)
            expansion_ratios.append(round(expansion, 2))

    expansion_ratios.sort()
    median_expansion = 1.0
    p90_expansion = 1.0
    if expansion_ratios:
        median_expansion = expansion_ratios[len(expansion_ratios) // 2]
        p90_idx = int(math.ceil(0.90 * len(expansion_ratios))) - 1
        p90_expansion = expansion_ratios[min(p90_idx, len(expansion_ratios) - 1)]

    # Behavioral label frequencies
    label_counts = {
        "already_good": sum(1 for r in accepted if r.labels.already_good),
        "compound_request": sum(1 for r in accepted if r.labels.compound_request),
        "self_correction": sum(1 for r in accepted if r.labels.self_correction),
        "ambiguous": sum(1 for r in accepted if r.labels.ambiguous),
        "negative_constraint": sum(1 for r in accepted if r.labels.negative_constraint),
    }

    sanitization_count = sum(1 for r in all_records if r.privacy.sanitized)

    # Diversity warnings
    warnings: List[str] = []
    if accepted:
        # Category domination
        for cat, cnt in category_counts.items():
            if cnt / len(accepted) > 0.40 and len(accepted) >= 10:
                warnings.append(f"Category '{cat}' dominates dataset ({cnt}/{len(accepted)} = {cnt/len(accepted):.1%})")

        # Excessive expansion
        if median_expansion > 6.0:
            warnings.append(f"High median prompt expansion ratio ({median_expansion:.1f}x). Check for prompt inflation.")

    stats = {
        "total_candidates": total,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate": round(len(accepted) / total, 4) if total > 0 else 0.0,
        "category_counts": category_counts,
        "source_counts": source_counts,
        "rejection_reasons": rejection_counts,
        "quality_distribution": {
            "mean": round(mean_quality, 2),
            "min": round(min_quality, 2),
            "max": round(max_quality, 2),
            "sample_size": len(quality_scores),
        },
        "expansion_metrics": {
            "median_expansion": median_expansion,
            "p90_expansion": p90_expansion,
        },
        "label_counts": label_counts,
        "sanitization_count": sanitization_count,
        "warnings": warnings,
    }
    return stats
