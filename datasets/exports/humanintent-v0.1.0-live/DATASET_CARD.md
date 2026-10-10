# HumanIntent Dataset Card: humanintent-v0.1.0-live

**Created At:** 2026-10-08T10:25:36.179350+00:00  
**Factory Version:** 0.1.0  
**Generator Model:** llama3  
**Random Seed:** 42  

## Purpose
This dataset is constructed by the HumanIntent Dataset Factory to train future intent-preserving prompt optimization models via Supervised Fine-Tuning (SFT) and Preference Optimization. It pairs informal, messy, or incomplete human prompts with grounded, validated, and quality-scored enhancements.

## Consent & Privacy Policy
- **Consented Real Data**: Sourced exclusively from local interaction history where `dataset_consent == True`. Unconsented interactions are strictly filtered at the SQL query level.
- **Privacy Sanitation**: All prompts undergo automated regex redaction of emails, phone numbers, API keys, bearer tokens, passwords, and private URLs. High-risk unredactable credentials trigger immediate record rejection.

## Dataset Statistics
- **Total Candidates Processed:** 22
- **Accepted Records:** 19
- **Rejected Records:** 3
- **Acceptance Rate:** 86.4%
- **Sanitized Records Count:** 0

### Category Coverage
- **CODING:** 2
- **DEBUGGING:** 2
- **WRITING:** 2
- **EDUCATION:** 1
- **RESEARCH:** 2
- **PLANNING:** 1
- **DATA_ANALYSIS:** 2
- **CREATIVE:** 2
- **IMAGE_GENERATION:** 1
- **SUMMARIZATION:** 2
- **GENERAL:** 2

### Quality & Expansion
- **Average Quality Score:** 94.47/100
- **Median Prompt Expansion:** 1.22x
- **P90 Prompt Expansion:** 3.24x

### Splits
- **Train Groups:** 17
- **Validation Groups:** 2

## Known Limitations
- Synthetic variants may exhibit stylistic regularities from the underlying LLM generator.
- Semantic embedding-based deduplication is deferred to future releases.
- Dataset generation is completely decoupled from model training (no fine-tuning was performed in Phase 9).
