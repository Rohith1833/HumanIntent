"""Configuration settings for Phase 10 benchmark runner."""

import os
from typing import List, Optional
from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Configuration options for formal benchmark execution."""
    benchmark_path: str = Field(
        default=os.path.join("benchmarks", "v1", "benchmark.jsonl"),
        description="Path to authoritative frozen benchmark JSONL",
    )
    metadata_path: str = Field(
        default=os.path.join("benchmarks", "v1", "metadata.json"),
        description="Path to benchmark metadata with SHA-256 digest",
    )
    output_dir: str = Field(
        default="evaluation_results",
        description="Root directory for run artifacts and error analysis",
    )
    model: str = Field(
        default="llama3:latest",
        description="Underlying LLM model tag for baselines and system",
    )
    judge_model: str = Field(
        default="llama3:latest",
        description="LLM used for blinded semantic judge",
    )
    temperature: float = Field(default=0.0, description="Deterministic temperature for evaluation")
    random_seed: int = Field(default=42, description="Global random seed for bootstrap and sampling")
    baselines: List[str] = Field(
        default_factory=lambda: ["raw", "generic", "humanintent"],
        description="Active baselines to evaluate",
    )
    bootstrap_iterations: int = Field(default=1000, description="Bootstrap iterations for confidence intervals")
    confidence_level: float = Field(default=0.95, description="Confidence interval level")
    limit: Optional[int] = Field(default=None, description="Optional cap on number of cases")
    sample_per_category: Optional[int] = Field(default=None, description="Sample N cases per category across all categories")
    categories: Optional[List[str]] = Field(default=None, description="Filter specific categories")
    difficulty: Optional[str] = Field(default=None, description="Filter specific difficulty")
    repeat: int = Field(default=1, description="Number of repeats for stochastic stability testing")
    dry_run: bool = Field(default=False, description="Validate schema and hash without making LLM calls")
    force: bool = Field(default=False, description="Overwrite existing cached predictions")
