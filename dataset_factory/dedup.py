import re
from typing import List, Set, Tuple
from dataset_factory.schemas import CurationStatus, DatasetRecord, RejectionReason


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def get_token_set(text: str) -> Set[str]:
    """Extract set of normalized tokens."""
    return set(normalize_text(text).split())


def token_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two texts."""
    tokens_a = get_token_set(text_a)
    tokens_b = get_token_set(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def deduplicate_records(
    records: List[DatasetRecord],
    threshold: float = 0.85,
) -> Tuple[List[DatasetRecord], List[DatasetRecord]]:
    """Deduplicate records using exact normalized text and near-lexical token Jaccard.

    Note: Semantic embedding-based deduplication is intentionally deferred to avoid
    premature vector dependencies.
    """
    accepted: List[DatasetRecord] = []
    duplicates: List[DatasetRecord] = []
    seen_normalized: Set[str] = set()

    for record in records:
        norm_prompt = normalize_text(record.original_prompt)

        # 1. Exact normalized match
        if norm_prompt in seen_normalized:
            record.curation_status = CurationStatus.REJECTED
            record.rejection_reasons.append(RejectionReason.DUPLICATE)
            duplicates.append(record)
            continue

        # 2. Near-duplicate lexical similarity against accepted pool
        is_near_dup = False
        for existing in accepted:
            # Only compare if lengths are comparable to save time
            if abs(len(record.original_prompt) - len(existing.original_prompt)) > 150:
                continue

            sim = token_jaccard_similarity(record.original_prompt, existing.original_prompt)
            if sim >= threshold:
                is_near_dup = True
                break

        if is_near_dup:
            record.curation_status = CurationStatus.REJECTED
            record.rejection_reasons.append(RejectionReason.DUPLICATE)
            duplicates.append(record)
        else:
            seen_normalized.add(norm_prompt)
            accepted.append(record)

    return accepted, duplicates
