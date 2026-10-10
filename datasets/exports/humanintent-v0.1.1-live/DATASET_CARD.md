# HumanIntent Dataset Card: humanintent-v0.1.1-live

**Created At:** 2026-10-08T15:53:23.258815+00:00  
**Factory Version:** 0.1.0  
**Generator Model:** llama3:latest  
**Random Seed:** 42  

## Purpose & Philosophy
This dataset is constructed by the HumanIntent Dataset Factory to train future intent-preserving prompt optimization models via Supervised Fine-Tuning (SFT) and Preference Optimization.

### Training-Data Acceptable != Runtime Acceptable
Teacher outputs are NEVER automatically trusted for training-data admission. Even when a teacher output achieves a high runtime quality score (>= 75) and passes runtime validation, it must independently pass the **Dataset Curation Gate**. The curation gate strictly enforces requirement recall, negative constraint preservation, quantity bounds, task-family consistency, structural well-formedness, and source dependency checks.

## Consent & Privacy Policy
- **Consented Real Data**: Sourced exclusively from local interaction history where `dataset_consent == True`. Unconsented interactions are strictly filtered at the SQL query level.
- **Privacy Sanitation**: All prompts undergo automated regex redaction of emails, phone numbers, API keys, bearer tokens, passwords, and private URLs. High-risk unredactable credentials trigger immediate record rejection.

## Dataset Statistics
- **Total Candidates Processed:** 22
- **Accepted Records:** 6
- **Rejected Records:** 16
- **Acceptance Rate:** 27.3%
- **Sanitized Records Count:** 0

### Curation Gate Breakdown
- **Target Integrity Pass Rate:** 27.3%
- **Source Dependency Rejections:** 0
- **Constraint Recall Failures:** 10
- **Malformed Target Rejections:** 2
- **Detailed Rejection Counts:**
  - `MISSING_CONSTRAINT`: 10
  - `MISSING_REQUIREMENT`: 7
  - `META_CONTENT`: 9
  - `QUANTITY_MISMATCH`: 3
  - `VALIDATION_FAILED`: 3
  - `MALFORMED_TARGET`: 2
  - `COMPOUND_TASK_LOSS`: 1

### Category Coverage
- **CODING:** 1
- **DEBUGGING:** 2
- **IMAGE_GENERATION:** 1
- **SUMMARIZATION:** 1
- **GENERAL:** 1

### Quality & Expansion
- **Average Quality Score:** 97.17/100
- **Median Prompt Expansion:** 1.44x
- **P90 Prompt Expansion:** 1.79x

### Splits
- **Train Groups:** 5
- **Validation Groups:** 1

## Known Limitations & Self-Distillation Risks
- **Self-Distillation Bias**: Training on teacher outputs risks inheriting latent stylistic biases and template phrasing from the teacher model.
- **Synthetic Stereotypes**: Synthetic variations can exhibit modal linguistic patterns; independent invariant gates reduce but do not eliminate this.
- **No Online Runtime Impact**: Curation is strictly an offline process; the runtime `/api/enhance` path remains untouched and lightweight.
- **No Training Conducted**: This phase constructs and hardens training data only; no model fine-tuning, LoRA, or embeddings have been applied.
