import random
from typing import Dict, List, Set, Tuple
from dataset_factory.schemas import DatasetRecord, SourceType


def split_records_by_group(
    records: List[DatasetRecord],
    train_ratio: float = 0.90,
    random_seed: int = 42,
) -> Tuple[List[DatasetRecord], List[DatasetRecord]]:
    """Split dataset records into Train and Validation sets grouped by seed/session.

    CRITICAL LEAKAGE INVARIANT: All variants belonging to the same SeedIntent
    (or all records from the same consented session) must remain strictly in the
    SAME split.
    """
    if not records:
        return [], []

    # 1. Group records by their parent seed_id or pseudonymized source_id
    grouped_records: Dict[str, List[DatasetRecord]] = {}
    for r in records:
        if r.source_type == SourceType.SYNTHETIC:
            group_key = r.provenance.seed_id or f"unbound_{r.record_id}"
        else:
            group_key = r.provenance.source_id or f"real_{r.record_id}"

        if group_key not in grouped_records:
            grouped_records[group_key] = []
        grouped_records[group_key].append(r)

    group_keys = sorted(list(grouped_records.keys()))
    rng = random.Random(random_seed)
    rng.shuffle(group_keys)

    # 2. Determine split boundary across groups
    n_groups = len(group_keys)
    if n_groups <= 1:
        # If only 1 group exists, put all in train
        train_groups: Set[str] = set(group_keys)
        val_groups: Set[str] = set()
    else:
        n_train = max(1, int(round(n_groups * train_ratio)))
        if n_train >= n_groups:
            n_train = n_groups - 1  # Guarantee at least 1 group in validation if possible

        train_groups = set(group_keys[:n_train])
        val_groups = set(group_keys[n_train:])

    # 3. Populate splits without row-level leakage
    train_records: List[DatasetRecord] = []
    val_records: List[DatasetRecord] = []

    for gk, recs in grouped_records.items():
        if gk in train_groups:
            train_records.extend(recs)
        else:
            val_records.extend(recs)

    return train_records, val_records
