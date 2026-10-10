"""
Authoritative loader for frozen HumanIntent benchmarks.
Verifies cryptographic SHA-256 integrity digest prior to loading cases.
"""

import hashlib
import json
import logging
import os
from typing import List, Optional

from evaluation.schemas import BenchmarkCase, BenchmarkMetadata

logger = logging.getLogger("evaluation.benchmark_loader")


class BenchmarkIntegrityError(Exception):
    """Raised when benchmark file is missing or has been tampered with."""
    pass


def _resolve_path(path: str) -> str:
    """Resolves path directly or relative to repository root."""
    if os.path.exists(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    return path


def verify_benchmark_integrity(benchmark_path: str, metadata_path: str) -> BenchmarkMetadata:
    """Computes SHA-256 of benchmark_path and asserts equality with metadata_path."""
    bm_path = _resolve_path(benchmark_path)
    meta_path = _resolve_path(metadata_path)

    if not os.path.exists(bm_path):
        raise BenchmarkIntegrityError(f"Benchmark file not found at: {benchmark_path}")
    if not os.path.exists(meta_path):
        raise BenchmarkIntegrityError(f"Metadata file not found at: {metadata_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    metadata = BenchmarkMetadata.model_validate(meta_dict)

    with open(bm_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    if actual_hash != metadata.sha256:
        raise BenchmarkIntegrityError(
            f"Benchmark integrity violation! Expected SHA-256 {metadata.sha256}, got {actual_hash}. "
            "Frozen benchmarks must never be mutated without a version bump."
        )

    return metadata


def load_benchmark(
    benchmark_path: str = os.path.join("benchmarks", "v1", "benchmark.jsonl"),
    metadata_path: str = os.path.join("benchmarks", "v1", "metadata.json"),
    categories: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    limit: Optional[int] = None,
    sample_per_category: Optional[int] = None,
) -> List[BenchmarkCase]:
    """
    Cryptographically verifies and loads benchmark cases with optional filtering.
    """
    # 1. Integrity check
    verify_benchmark_integrity(benchmark_path, metadata_path)
    bm_path = _resolve_path(benchmark_path)

    cases: List[BenchmarkCase] = []
    cat_counts: dict = {}
    with open(bm_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            case = BenchmarkCase.model_validate_json(line)

            # Filtering
            if categories and case.category not in categories:
                continue
            if difficulty and case.difficulty.value.upper() != difficulty.upper():
                continue
            if sample_per_category is not None:
                cur = cat_counts.get(case.category, 0)
                if cur >= sample_per_category:
                    continue
                cat_counts[case.category] = cur + 1

            cases.append(case)
            if limit and len(cases) >= limit:
                break

    logger.info("Loaded %d validated benchmark cases from %s", len(cases), benchmark_path)
    return cases
