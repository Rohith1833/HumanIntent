# HumanIntent Phase 10 Evaluation & Benchmarking Framework

## Overview
Phase 10 introduces a formal, reproducible evaluation harness and frozen benchmark for **HumanIntent — An Intent-Preserving Adaptive Prompt Optimization System**.

The framework strictly guarantees that:
- **No model training occurs:** Models are evaluated frozen as-is against Ollama `llama3:latest`.
- **Zero production runtime patching:** Production runtime modules (`app.core.*`, `app.api.routes`) are not modified or tuned to overfit benchmark results.
- **Cryptographic Benchmark Integrity:** The benchmark file is cryptographically locked via SHA-256 digest in `benchmarks/v1/metadata.json`. Any runtime modification immediately halts evaluation.
- **No Benchmark Data Contamination:** Benchmark cases are completely separated from Phase 9 seed prompts and existing test cases.

---

## Directory Architecture

```
evaluation/
    __init__.py             # Exports schemas, config, loader, runner, metrics
    schemas.py              # Pydantic schemas (BenchmarkCase, SemanticJudgeResult, etc.)
    config.py               # EvaluationConfig dataclass/pydantic model
    benchmark_loader.py     # SHA-256 verified loader with category/difficulty filtering
    runner.py               # Asynchronous orchestrator with checkpointing & resume
    metrics.py              # Invariant metrics, confusion matrix, and release gates
    baselines.py            # Baselines A (RAW), B (GENERIC), C (ABLATION), D (HUMANINTENT)
    judges.py               # Independent semantic judge & blinded pairwise A/B comparison
    bootstrap.py            # Non-parametric bootstrap resampling for 95% CIs
    stability.py            # Multi-run repeatability & variance analysis
    report.py               # Markdown and machine-readable JSON report generation
    README.md               # Framework documentation

benchmarks/
    v1/
        benchmark.jsonl     # 220 frozen benchmark prompts (20 per category across all 11)
        metadata.json       # SHA-256 hash digest, schema version, creation timestamp
        rubric.md           # Evaluation rubric and invariant definitions

evaluation_results/
    <run_id>/
        predictions.jsonl   # Case predictions across all evaluated baselines
        metrics.json        # Machine-readable evaluation metrics and gate statuses
        report.md           # Formal evaluation report with executive summary
        confusion_matrix.json # Category confusion matrix
        error_analysis.jsonl  # Case-level failure taxonomy and inspector logs

scripts/
    run_evaluation.py       # CLI runner for local evaluation
    inspect_evaluation.py   # CLI inspector for error analysis and metrics inspection
```

---

## Benchmark Composition (`humanintent-benchmark-v1.0`)

The frozen benchmark comprises **220 cases** spanning all 11 canonical `PromptCategory` values:
1. `CODING`: 20 cases
2. `DEBUGGING`: 20 cases
3. `WRITING`: 20 cases
4. `EDUCATION`: 20 cases
5. `RESEARCH`: 20 cases
6. `PLANNING`: 20 cases
7. `DATA_ANALYSIS`: 20 cases
8. `CREATIVE`: 20 cases
9. `IMAGE_GENERATION`: 20 cases
10. `SUMMARIZATION`: 20 cases
11. `GENERAL`: 20 cases

### Difficulty Distribution
- **EASY:** ~34% (74 cases)
- **MEDIUM:** ~40% (89 cases)
- **HARD:** ~26% (57 cases)

### Behavioral Coverage
The benchmark specifically probes:
- Invariant requirement preservation (`must_preserve`)
- Anti-hallucination / anti-invention constraints (`must_not_invent`)
- Explicit negative constraints (`no third-party dependencies`, `no recursion`)
- Exact numerical quantities (`3 options`, `150 words`, `4 weeks`)
- Source dependency discrimination (missing context vs. self-contained)
- Ambiguity detection and clarification triggering
- Adversarial task-family confusion and meta-instruction leakage probes

---

## Evaluated Baselines
1. **Baseline A (`raw`):** Raw user prompt unchanged.
2. **Baseline B (`generic`):** Generic prompt enhancement prompt (`"Act as an expert prompt engineer..."`) executed directly on `llama3:latest` without structured analysis.
3. **Baseline C (`no_validator_ablation`):** HumanIntent pipeline executed with validation/repair bypassed to isolate the impact of the validator.
4. **Baseline D (`humanintent`):** Full end-to-end HumanIntent pipeline.

---

## Release Gates (Phase 10 Starting Thresholds)

| Release Gate Metric | Target Threshold | Description |
|---|---|---|
| **Category Accuracy** | $\ge 90.0\%$ | Macro category classification accuracy |
| **Task-Family Accuracy** | $\ge 95.0\%$ | IntentAnchor action family classification |
| **Explicit Requirement Recall** | $\ge 95.0\%$ | Percentage of `must_preserve` invariants retained |
| **Constraint Recall** | $\ge 95.0\%$ | Negative and structural constraint preservation |
| **Quantity Preservation** | $\ge 97.0\%$ | Exact retention of numerical limits and quantities |
| **Unsupported Requirement Rate** | $\le 3.0\%$ | Inventions / hallucinations introduced |
| **Objective Drift Rate** | $\le 2.0\%$ | Undesired changes in core user goal |
| **Clarification F1** | $\ge 90.0\%$ | Harmonic mean of precision & recall on ambiguous inputs |
| **Over-Clarification Rate** | $\le 5.0\%$ | Percentage of clear inputs erroneously gated |
| **Meta / Instruction Leakage** | $== 0.0\%$ | Zero system control prompts leaked into user output |

---

## Running Evaluations

### 1. Dry Run (Integrity & Schema Verification)
```bash
python scripts/run_evaluation.py --dry-run
```

### 2. Pilot / Sampled Evaluation
```bash
python scripts/run_evaluation.py --limit 22 --output evaluation_results/pilot_001
```

### 3. Stratified Multi-Category Evaluation
```bash
python scripts/run_evaluation.py --sample-per-category 3 --output evaluation_results/stratified_001
```

### 4. Inspect Results & Error Analysis
```bash
python scripts/inspect_evaluation.py --run-dir evaluation_results/pilot_001 --show-failures
```
