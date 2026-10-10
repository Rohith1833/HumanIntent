import json
import os
from typing import Dict, List, Set
from dataset_factory.schemas import SeedIntent


CANONICAL_CATEGORIES = {
    "CODING",
    "DEBUGGING",
    "WRITING",
    "EDUCATION",
    "RESEARCH",
    "PLANNING",
    "DATA_ANALYSIS",
    "CREATIVE",
    "IMAGE_GENERATION",
    "SUMMARIZATION",
    "GENERAL",
}


def load_seeds(seeds_path: str) -> List[SeedIntent]:
    """Load and validate seed intents from a JSONL file."""
    if not os.path.exists(seeds_path):
        raise FileNotFoundError(f"Seeds file not found at '{seeds_path}'")

    seeds: List[SeedIntent] = []
    seen_ids: Set[str] = set()

    with open(seeds_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_num} of '{seeds_path}': {exc}") from exc

            seed = SeedIntent.model_validate(data)

            if seed.seed_id in seen_ids:
                raise ValueError(f"Duplicate seed_id '{seed.seed_id}' found on line {line_num} of '{seeds_path}'")

            seen_ids.add(seed.seed_id)
            seeds.append(seed)

    return seeds


def get_category_distribution(seeds: List[SeedIntent]) -> Dict[str, int]:
    """Compute per-category counts across loaded seeds."""
    dist: Dict[str, int] = {cat: 0 for cat in sorted(CANONICAL_CATEGORIES)}
    for seed in seeds:
        cat = seed.category
        dist[cat] = dist.get(cat, 0) + 1
    return dist


def save_seeds(seeds: List[SeedIntent], output_path: str) -> None:
    """Save seed intents to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seed in seeds:
            f.write(seed.model_dump_json() + "\n")
