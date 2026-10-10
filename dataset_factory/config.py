import json
import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

try:
    import yaml
except ImportError:
    yaml = None


class DatasetFactoryConfig(BaseModel):
    """Configuration settings for the Dataset Factory pipeline."""
    dataset_version: str = Field("humanintent-v0.1.0", description="Semantic dataset version tag")
    random_seed: int = Field(42, description="Global random seed for deterministic operations")
    generator_model: str = Field("llama3", description="Model used for synthetic variant generation")
    generator_temperature: float = Field(0.7, description="Sampling temperature for synthetic generation")
    synthetic_per_seed: int = Field(3, description="Target number of variants per seed intent")
    include_consented_real: bool = Field(False, description="Whether to ingest consented real interaction history")
    synthetic_only: bool = Field(True, description="Generate exclusively synthetic data")
    real_only: bool = Field(False, description="Ingest exclusively consented real data")
    minimum_quality_score: float = Field(75.0, description="Minimum overall quality score for positive SFT")
    max_inflation_ratio: float = Field(8.0, description="Maximum allowed length inflation before rejection")
    near_duplicate_threshold: float = Field(0.85, description="Jaccard similarity threshold for near-duplicate rejection")
    train_split_ratio: float = Field(0.90, description="Proportion of seed groups allocated to train split")
    val_split_ratio: float = Field(0.10, description="Proportion of seed groups allocated to validation split")
    review_sample_size: int = Field(50, description="Number of accepted records in stratified human review sample")
    output_dir: str = Field("datasets/exports", description="Destination directory for exported datasets")
    cache_dir: str = Field("datasets/cache", description="Directory for caching LLM generation and teacher evaluation")
    seeds_path: str = Field("datasets/seeds/v1.jsonl", description="Path to seed intents JSONL file")
    database_url: Optional[str] = Field(None, description="Database URL for Phase 8 SQLite history (optional override)")
    dry_run: bool = Field(False, description="Dry run mode: validate plan without calling LLM or writing exports")
    force: bool = Field(False, description="Force re-generation ignoring existing caches")


def load_config(config_path: Optional[str] = None, **overrides: Any) -> DatasetFactoryConfig:
    """Load configuration from a YAML/JSON file and apply keyword overrides."""
    data: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith((".yaml", ".yml")) and yaml:
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)

    # Clean None values from overrides so defaults aren't overwritten
    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
    data.update(clean_overrides)

    return DatasetFactoryConfig(**data)
