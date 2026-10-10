# Phase 10 Formal Evaluation Report: eval_humanintent

**Benchmark Version:** humanintent-benchmark-v1.0  
**Model Evaluated:** llama3:latest  
**Total Benchmark Cases Evaluated:** 220 / 220  

## Executive Summary
This report presents the formal evaluation of the HumanIntent prompt optimization system against the frozen `humanintent-benchmark-v1.0` benchmark across all 11 canonical categories.

## Release Gate Status

| Metric Gate | Target Threshold | Measured Actual | Gate Status |
|---|---|---|---|
| **Category Accuracy** | >= 90.0% | 60.38% | FAILED |
| **Task-Family Accuracy** | >= 95.0% | 60.38% | FAILED |
| **Explicit Requirement Recall** | >= 95.0% | 92.95% | FAILED |
| **Constraint Recall** | >= 95.0% | 73.44% | FAILED |
| **Quantity Preservation** | >= 97.0% | 86.09% | FAILED |
| **Unsupported Requirement Rate** | <= 3.0% | 10.91% | FAILED |
| **Objective Drift Rate** | <= 2.0% | 0.0% | PASSED |
| **Clarification F1** | >= 90.0% | 0.0% | FAILED |
| **Over-Clarification Rate** | <= 5.0% | 0.0% | PASSED |
| **Meta / Instruction Leakage** | == 0.0% | 19.09% | FAILED |

## Baseline Performance Comparison

| Evaluation Metric | Baseline A (RAW) | Baseline B (GENERIC) | Baseline D (HUMANINTENT) |
|---|---|---|---|
| Category Accuracy | N/A | N/A | **60.4%** |
| Task-Family Accuracy | N/A | N/A | **60.4%** |
| Explicit Requirement Recall | 98.1% | 92.4% | **93.0%** |
| Constraint Recall | 95.8% | 84.5% | **73.4%** |
| Quantity Preservation | 100.0% | 70.5% | **86.1%** |
| Unsupported Requirement Rate | 9.6% | 16.4% | **10.9%** |
| Clarification F1 | 0.0% | 0.0% | **0.0%** |
| Over-Clarification Rate | 0.0% | 0.0% | **0.0%** |
| Median Expansion Ratio | 1.00x | 3.33x | **3.50x** |
| Median Latency | 0.0 ms | 8352.0 ms | **97508.8 ms** |
| Average LLM Calls | 0.0 | 1.0 | **4.43** |

## 95% Bootstrap Confidence Intervals (Non-Parametric Resampling)

| Primary Evaluation Metric | Measured Point Estimate | 95% Bootstrap CI | Bootstrap Resample Mean | Estimator Status |
|---|---|---|---|---|
| **Category Accuracy** | 60.38% | [53.81%, 66.82%] | 60.37% | Aligned |
| **Task-Family Accuracy** | 60.38% | [53.81%, 66.82%] | 60.37% | Aligned |
| **Explicit Requirement Recall** | 92.95% | [90.49%, 95.37%] | 93.00% | Aligned |
| **Constraint Recall** | 73.44% | [61.90%, 83.33%] | 73.58% | Aligned |
| **Quantity Preservation** | 86.09% | [75.97%, 95.33%] | 86.35% | Aligned |
| **Unsupported Requirement Rate** | 10.91% | [6.82%, 15.00%] | 10.86% | Aligned |
| **Clarification F1** | 0.00% | [0.00%, 0.00%] | 0.00% | Aligned |

## Call Accounting & Stage Budgets
- **Average LLM Calls per Case:** 4.43 (Median: 4.0, P90: 6.0, P95: 6.0)
- **Clear Enhanced Path Calls:** 4.69 (Target: 4 calls [1 analyzer + 1 enhancer + 1 validator + 1 scorer])
- **Clarification First-Pass Calls:** 0.00 (Target: 1 call [1 analyzer])
- **Analysis Repair Cases:** 26
- **Enhancement Repair Cases:** 61

## Stage Latency Distribution

| Stage | Median (P50) | P90 | P95 |
|---|---|---|---|
| Analysis | 34655.1 ms | 68018.1 ms | 83502.6 ms |
| Enhancement | 17944.2 ms | 25659.8 ms | 27972.8 ms |
| Repair | 22183.8 ms | 35829.9 ms | 39920.3 ms |
| Scoring | 31605.6 ms | 39309.7 ms | 41062.3 ms |
| Total | 97508.8 ms | 144738.1 ms | 163690.3 ms |
| Validation | 3341.5 ms | 27148.4 ms | 32728.5 ms |

## Per-Category Performance Breakdown

| Category | Cases | Category Acc | Req Recall | Const Recall | Qty Pres | Unsupp Rate | Expansion |
|---|---|---|---|---|---|---|---|
| `CODING` | 20 | 95.0% | 90.7% | 69.2% | 100.0% | 15.0% | 3.70x |
| `CREATIVE` | 20 | 35.0% | 90.2% | 72.7% | 91.7% | 10.0% | 4.23x |
| `DATA_ANALYSIS` | 20 | 55.0% | 88.2% | 100.0% | 75.0% | 10.0% | 3.33x |
| `DEBUGGING` | 20 | 35.0% | 82.0% | 100.0% | 75.0% | 15.0% | 3.84x |
| `EDUCATION` | 20 | 95.0% | 82.4% | 25.0% | 85.7% | 15.0% | 3.49x |
| `GENERAL` | 20 | 35.0% | 95.5% | 85.7% | 100.0% | 15.0% | 4.23x |
| `IMAGE_GENERATION` | 20 | 40.0% | 98.8% | 100.0% | 100.0% | 10.0% | 3.66x |
| `PLANNING` | 20 | 65.0% | 91.4% | 100.0% | 93.1% | 10.0% | 4.89x |
| `RESEARCH` | 20 | 55.0% | 90.1% | 100.0% | 100.0% | 0.0% | 3.79x |
| `SUMMARIZATION` | 20 | 80.0% | 62.9% | 47.4% | 48.0% | 10.0% | 1.15x |
| `WRITING` | 20 | 50.0% | 89.3% | 62.5% | 72.7% | 10.0% | 3.00x |

## Per-Difficulty Performance Breakdown

| Difficulty | Cases | Category Acc | Req Recall | Const Recall | Qty Pres | Unsupp Rate | Expansion |
|---|---|---|---|---|---|---|---|
| `EASY` | 74 | 62.2% | 80.2% | 62.5% | 86.1% | 6.8% | 3.66x |
| `HARD` | 57 | 54.4% | 85.8% | 70.4% | 65.8% | 10.5% | 3.69x |
| `MEDIUM` | 89 | 57.3% | 94.3% | 65.0% | 88.9% | 14.6% | 3.39x |

## Blinded Pairwise Win Rates (Explicit Denominators)
- **HumanIntent vs. RAW Prompt:** 0 / 22 wins (0.0%) | Loss: 0.0% | Tie: 100.0%
- **HumanIntent vs. GENERIC Enhancer:** 0 / 22 wins (0.0%) | Loss: 0.0% | Tie: 100.0%
- **Position Bias Consistency:** 22 / 22 cases consistent after swap (100.0%)

## Clarification & Source Dependency
- **Clarification Precision:** 100.0%
- **Clarification Recall:** 0.0%
- **Clarification F1 Score:** 0.0%
- **Over-Clarification Rate:** 0.0%
- **Under-Clarification Rate:** 100.0%
- **Source Dependency Accuracy:** 98.6%

## Top Failure Classes
- `CATEGORY_ERROR`: 84 occurrences
- `TASK_FAMILY_ERROR`: 84 occurrences
- `META_LEAKAGE`: 42 occurrences
- `MISSING_REQUIREMENT`: 38 occurrences
- `INVENTED_REQUIREMENT`: 24 occurrences
- `MISSING_CONSTRAINT`: 17 occurrences
- `UNDER_CLARIFICATION`: 13 occurrences
- `QUANTITY_ERROR`: 10 occurrences
- `SOURCE_DEPENDENCY_ERROR`: 3 occurrences

## Known Limitations & Empirical Justification for Phase 11 Training
1. **Unrequested Feature Invention**: Zero-shot base models naturally embellish prompts with unrequested defaults.
2. **Negative Constraint Erosion**: Strict negative constraints ('no recursion', 'no external libraries') are occasionally softened by base models.
3. **Zero Model Training Performed**: All results evaluate the zero-shot prompting pipeline without any fine-tuning.
