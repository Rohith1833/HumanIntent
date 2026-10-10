# HumanIntent Dataset Factory

The **HumanIntent Dataset Factory** is an offline, production-oriented infrastructure designed to build clean, versioned, and reproducible training datasets for future fine-tuning.

> **CRITICAL ARCHITECTURAL PRINCIPLE:**
> **Dataset Generation != Model Training.**
> Phase 9 builds the data generation, curation, and export pipeline. It does **not** train any model, execute fine-tuning, implement LoRA/QLoRA, or introduce vector databases.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Curated Seed Intents      |
                      |   (datasets/seeds/v1.jsonl) |
                      +--------------+--------------+
                                     |
                                     v
                       +---------------------------+
                       | Synthetic Variant Gen     |
                       | (Ollama / 9 Variant Types)|
                       +-------------+-------------+
                                     |
+--------------------------+         |
| Consented Real History   |         |
| (dataset_consent == True)|         |
+------------+-------------+         |
             |                       |
             v                       v
+--------------------------------------------------+
|           Privacy & Credential Sanitizer          |
|      (Masks PII/Tokens; Rejects High-Risk Keys)   |
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
|           HumanIntent Teacher Evaluation         |
|      (Analyzer -> Enhancer -> Validator -> Score)|
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
|              Staged Filter Pipeline              |
| (Grounding, Validation, Quality >= 75, Inflation)|
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
|          Lexical Deduplication Engine            |
|       (Exact Token Hash + Jaccard >= 0.85)       |
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
|          Group-Based Dataset Splitter            |
|     (Seed/Session Grouped: 90% Train / 10% Val)  |
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
|                 Dataset Exporter                 |
| - sft_{train,val}.jsonl                          |
| - clarification_{train,val}.jsonl                |
| - preference_{train,val}.jsonl                   |
| - review_sample.jsonl                            |
| - manifest.json & DATASET_CARD.md                |
+--------------------------------------------------+
```

---

## Key Design Principles & Safeguards

### 1. Consent Policy (Zero Inferred Consent)
- **Local History != Dataset Consent**: User prompt history stored in SQLite is strictly private by default.
- Real interactions are **only** eligible for dataset extraction if `dataset_consent == True`.
- Helpful ratings or user edits are never treated as implicit consent.
- Filtering is enforced directly at the SQL query level before records are ever loaded into candidate buffers.
- Exported records use pseudonymized, one-way hashed identifiers (`real_<hash>`). Raw session UUIDs are never published.

### 2. Privacy & Secret Sanitation
- Automated regex sanitizer redacts emails (`<EMAIL>`), phone numbers (`<PHONE>`), API keys/tokens (`<API_KEY>`), passwords (`<PASSWORD>`), authenticated URLs (`<URL>`), and personal IDs (`<ID>`).
- If an unredactable private cryptographic key block (`-----BEGIN ... PRIVATE KEY-----`) is detected, the candidate is immediately excluded with `RejectionReason.PRIVACY_RISK`.

### 3. Variant Taxonomy
The synthetic generator derives 9 realistic human prompt variations from each seed:
1. `CLEAN`: Direct, well-structured request.
2. `CASUAL`: Conversational, informal phrasing.
3. `MESSY`: Rambling, stream-of-consciousness input with filler words.
4. `SHORT`: Minimal, telegraphic specification.
5. `COMPOUND`: Multiple related deliverables combined.
6. `SELF_CORRECTING`: User corrects or alters a requirement mid-prompt.
7. `CONSTRAINED`: Explicit negative or quantitative boundaries.
8. `ALREADY_GOOD`: Exemplary prompt requiring minimal or zero alteration.
9. `AMBIGUOUS`: Crucial details missing to test clarification flows.

### 4. Zero Train/Val Leakage (Group Splitting)
Splitting is executed by **Seed / Session Group**, never by random row sampling:
- All 9 variants derived from `coding_001` are assigned strictly to `train` or `validation`.
- Cross-split contamination and evaluation leakage are mathematically prevented.

### 5. Lexical Deduplication
- Exact normalized text matching.
- Token-level Jaccard similarity threshold (default `0.85`).
- Vector embeddings and semantic clustering are intentionally deferred to future benchmarking phases.

---

## Directory Structure

```
dataset_factory/
    __init__.py              # Package init
    schemas.py               # Pydantic schemas (DatasetRecord, SFTExample, etc.)
    config.py                # Pipeline configuration
    seed_loader.py           # Seed intent loader & validator
    synthetic_generator.py   # 9-variant prompt generator with caching
    real_data_loader.py      # Consented Phase 8 SQLite history loader
    sanitizer.py             # PII & credential redaction
    candidate_builder.py     # Candidate record assembler
    evaluator.py             # Locked HumanIntent teacher evaluator
    filters.py               # Staged quality & safety filters
    dedup.py                 # Exact & near-lexical deduplication
    splitter.py              # Leak-free group splitter
    exporter.py              # JSONL, manifest & dataset card exporter
    stats.py                 # Dataset metrics & diversity checks
    pipeline.py              # Orchestration coordinator

datasets/
    seeds/
        v1.jsonl             # 110 curated seeds across all 11 categories
    raw/                     # Raw generated variants
    intermediate/            # Resumable candidate checkpoints
    exports/                 # Versioned output packages (e.g. humanintent-v0.1.0/)

scripts/
    generate_dataset.py      # Build CLI
    inspect_dataset.py       # Dataset audit & inspection CLI
```

---

## CLI Usage

### Build Dataset
```bash
# Dry run: check plan and seed coverage without executing LLM calls
python scripts/generate_dataset.py --dry-run

# Standard synthetic build
python scripts/generate_dataset.py \
  --version humanintent-v0.1.0 \
  --seed datasets/seeds/v1.jsonl \
  --synthetic-per-seed 3 \
  --generator-model llama3

# Include consented real history
python scripts/generate_dataset.py \
  --version humanintent-v0.1.0 \
  --include-consented-real
```

### Inspect Dataset Export
```bash
# Inspect generated dataset manifest and review sample
python scripts/inspect_dataset.py --version humanintent-v0.1.0
```

---

## Known Limitations
- Semantic vector-based deduplication is deferred.
- Synthetic prompts reflect the stylistic tendencies of local generation models (`llama3`).
- Datasets are formatted for future fine-tuning, but training is strictly out of scope for Phase 9.
