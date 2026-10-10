# Phase 10 Formal Evaluation Report: eval_humanintent

**Benchmark Version:** humanintent-benchmark-v1.0  
**Model Evaluated:** llama3:latest  
**Total Benchmark Cases Evaluated:** 11 / 11  

## Executive Summary
This report presents the formal evaluation of the HumanIntent prompt optimization system against the frozen `humanintent-benchmark-v1.0` benchmark. The system was evaluated across all 11 canonical categories and multiple baseline conditions without modifying production runtime code.

## Release Gate Status

| Metric Gate | Target Threshold | Measured Actual | Gate Status |
|---|---|---|---|
| **Category Accuracy** | >= 90.0% | 90.91% | PASSED |
| **Task-Family Accuracy** | >= 95.0% | 90.91% | FAILED |
| **Explicit Requirement Recall** | >= 95.0% | 96.88% | PASSED |
| **Constraint Recall** | >= 95.0% | 80.0% | FAILED |
| **Quantity Preservation** | >= 97.0% | 80.0% | FAILED |
| **Unsupported Requirement Rate** | <= 3.0% | 9.09% | FAILED |
| **Objective Drift Rate** | <= 2.0% | 0.0% | PASSED |
| **Clarification F1** | >= 90.0% | 100.0% | PASSED |
| **Over-Clarification Rate** | <= 5.0% | 0.0% | PASSED |
| **Meta / Instruction Leakage** | == 0.0% | 0.0% | PASSED |

## Baseline Performance Comparison

| Evaluation Metric | Baseline A (RAW) | Baseline B (GENERIC) | Baseline D (HUMANINTENT) |
|---|---|---|---|
| Category Accuracy | N/A | N/A | **90.9%** |
| Task-Family Accuracy | N/A | N/A | **90.9%** |
| Explicit Requirement Recall | 100.0% | 100.0% | **96.9%** |
| Constraint Recall | 100.0% | 100.0% | **80.0%** |
| Quantity Preservation | 100.0% | 100.0% | **80.0%** |
| Unsupported Requirement Rate | 0.0% | 0.0% | **9.1%** |
| Clarification F1 | 0.0% | 0.0% | **100.0%** |
| Over-Clarification Rate | 0.0% | 0.0% | **0.0%** |
| Median Expansion Ratio | 1.00x | 1.00x | **3.97x** |
| Median Latency | 0.0 ms | 0.0 ms | **87479.3 ms** |
| Average LLM Calls | 0.0 | 1.0 | **2.00** |

## Blinded Pairwise Win Rates
- **HumanIntent vs. RAW Prompt:** Win: 100.0% | Loss: 0.0% | Tie: 0.0%
- **HumanIntent vs. GENERIC Enhancer:** Win: 100.0% | Loss: 0.0% | Tie: 0.0%

## Clarification & Source Dependency
- **Clarification Precision:** 100.0%
- **Clarification Recall:** 100.0%
- **Clarification F1 Score:** 100.0%
- **Over-Clarification Rate:** 0.0%
- **Under-Clarification Rate:** 0.0%
- **Source Dependency Accuracy:** 100.0%

## Top Failure Classes
- `INVENTED_REQUIREMENT`: 1 occurrences
- `QUANTITY_ERROR`: 1 occurrences
- `MISSING_CONSTRAINT`: 1 occurrences
- `MISSING_REQUIREMENT`: 1 occurrences
- `CATEGORY_ERROR`: 1 occurrences
- `TASK_FAMILY_ERROR`: 1 occurrences

## Known Limitations & Findings for Phase 11
1. **Template Framing Rigidity**: In certain categories, category template headers contribute fixed structural tokens.
2. **Ambiguity Calibration**: Short incomplete requests benefit significantly from clarification, while compound requests occasionally risk over-clarification.
3. **Zero Model Training**: All evaluations measure the zero-shot prompting pipeline; no fine-tuning was performed.
