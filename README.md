# HumanIntent

> **Intent-Preserving Adaptive Prompt Optimization System**

HumanIntent translates informal, incomplete, or ambiguous human requests into clear, high-quality, structured prompts for AI systems—**without silently inventing unrequested technologies, frameworks, or facts, and without making prompts arbitrarily long.**

---

## Current Status: Phase 8 — Local Persistence, History & Feedback Foundation

This repository implements **Phase 8**:
- **Local SQLite Persistence Layer**: Local-first storage powered by SQLAlchemy 2.x and Alembic migrations. Stores session history, authoritative user inputs, versioned analysis snapshots, enhancement attempts (initial + repair), validation snapshots, quality reports, and user feedback.
- **Core Privacy Principle: Local History != Training Consent**:
  - Storing a prompt locally for history does **NOT** grant permission for dataset construction, training, or evaluation corpora export.
  - Dataset consent is **explicit**, **separate**, and **defaults strictly to FALSE**.
  - Local history works completely independently of training consent.
- **Core Data Principle: Authoritative vs Derived Separation**:
  - The raw user prompt and clarification answers are authoritative human inputs and are **never overwritten**.
  - Analyses, generated prompts, repairs, validations, and quality reports are stored separately as versioned audit snapshots.
  - User-edited prompts submitted in feedback are treated as user feedback, not canonical system truth.
- **Strict Zero-LLM History Operations**:
  - Listing history (`GET /api/history`), loading historical sessions (`GET /api/history/{id}`), deleting history (`DELETE /api/history/{id}`), and submitting feedback (`POST /api/feedback`) consume **zero LLM calls**.
  - History summaries use deterministic prompt previews (first 100 characters).
- **Graceful Persistence Failure**:
  - AI generation and validation semantics remain logically intact even if a database write error occurs.
  - SQLite transactions are short and never held open during Ollama inference.
- **Configurable History**:
  - Controlled via `HISTORY_ENABLED=true` (default: true). If disabled, the system operates in ephemeral mode without recording sessions.

---

## Architecture Flow (Phase 7)

```text
Browser (User Input)
       │
       ▼
Next.js Frontend (http://localhost:3000)
       │  POST /api/enhance
       ▼
FastAPI Backend (http://localhost:8000)
       │
       ├─► 1. PromptAnalyzer (ONE structured analysis LLM call with JSON contract)
       │       └── Extracts classification, objective, requirements, constraints,
       │           preferences, missing_context, and ambiguity assessment
       │
       ├─► 2. Ambiguity Decision Gate
       │       │
       │       ├─► IF needs_clarification == true AND NOT allow_reasonable_assumptions:
       │       │       └── Return { status: "needs_clarification", questions: [...] }
       │       │           (0 enhancer calls, 0 validator calls, 0 scorer calls)
       │       │
       │       └─► IF needs_clarification == false OR allow_reasonable_assumptions == true:
       │               │
       │               ├─► Deterministic Template Selector (O(1) Python lookup)
       │               │
       │               ├─► 3. PromptEnhancer (Initial enhancement generation)
       │               │
       │               ├─► 4. PromptValidator (Hybrid deterministic + structured LLM check)
       │               │       │
       │               │       ├─► VALID (0 errors):
       │               │       │       │
       │               │       │       └─► 5. PromptScorer (ONE structured scoring LLM call)
       │               │       │               └── Return { status: "enhanced", quality: {...}, validation: {...} }
       │               │       │
       │               │       └─► INVALID (>= 1 error):
       │               │               │
       │               │               ├─► PromptEnhancer Repair (ONE controlled repair attempt)
       │               │               │
       │               │               ├─► Re-validation
       │               │               │       │
       │               │               │       ├─► REPAIR VALID:
       │               │               │       │       └─► PromptScorer (ONE structured scoring LLM call)
       │               │               │       │               └── Return { status: "enhanced", quality: {...} }
       │               │               │       │
       │               │               │       └─► REPAIR INVALID:
       │               │               │               └── Return { status: "validation_failed" } (0 scorer calls)
       │
       ▼
Frontend UI:
   - If enhanced: Renders Understood Intent + Enhanced Prompt + Subtle "Intent checked ✓" badge + Compact "Prompt Quality Estimate" breakdown
   - If needs_clarification: Renders Understood Intent + Clarification Card
   - If validation_failed: Renders friendly notice + "Try Again" action (no raw JSON dump)
```

---

## Prerequisites

1. **Python 3.10+**
2. **Node.js 18+** & **npm**
3. **Ollama**: Installed locally ([ollama.com](https://ollama.com))
4. **Local Llama Model**:
   ```powershell
   ollama run llama3
   # or fallback
   ollama run llama3.2
   ```

---

## Quick Start (Windows / PowerShell)

### 1. Automated Setup
Run the setup script from the repository root:
```powershell
.\setup.ps1
```
This checks prerequisites, initializes the Python virtual environment, installs backend dependencies, and installs frontend npm packages.

### 2. Ensure Ollama is Running
Make sure Ollama is running and has the model pulled:
```powershell
ollama serve
# In another terminal:
ollama pull llama3
```

### 3. Launch Development Servers
Run the dev launcher:
```powershell
.\dev.ps1
```
This starts both the FastAPI backend (`http://localhost:8000`) and Next.js frontend (`http://localhost:3000`) in separate terminal windows.

---

## Manual Execution (Terminal by Terminal)

### Backend
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`
- Analyze endpoint: `POST http://localhost:8000/api/analyze`
- Enhance endpoint: `POST http://localhost:8000/api/enhance`
- History list: `GET http://localhost:8000/api/history?limit=20&offset=0`
- History detail: `GET http://localhost:8000/api/history/{session_id}`
- Delete history: `DELETE http://localhost:8000/api/history/{session_id}`
- Clear history: `DELETE http://localhost:8000/api/history`
- Feedback submit: `POST http://localhost:8000/api/feedback`

### Frontend
```powershell
cd frontend
npm install
npm run dev
```
- App UI: `http://localhost:3000`

---

## Configuration & Environment Variables

### Backend (`backend/.env`)
The persistence layer and LLM configurations are fully environment-driven:
```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=120
BACKEND_CORS_ORIGINS=["http://localhost:3000"]

# Phase 8 Persistence
HUMANINTENT_DATABASE_URL=sqlite:///./data/humanintent.db
HISTORY_ENABLED=true
```

> **Privacy Note**: `Stored locally` does NOT mean `training consent`. Consent defaults strictly to `false` and is explicitly tracked on feedback submissions.

### Frontend (`frontend/.env.local`)
```ini
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## Running Tests

### Backend Unit, Persistence, Boundary & Coherence Tests (264 Tests)
Automated tests mock the Ollama client layer, running fast without requiring a live Ollama instance:
```powershell
cd backend
python -m pytest tests -v
```

### Database Migrations
```powershell
cd backend
alembic upgrade head
```

### Frontend Build Validation
```powershell
cd frontend
npm run build
```

---

## Semantic Safeguards: Source Grounding, Semantic Grounding & Cross-Field Coherence

HumanIntent enforces three distinct, non-overlapping safeguards across the analysis and enhancement pipeline:

### 1. SOURCE GROUNDING
> *"Did this information come from the user?"*

- **Purpose**: Verifies provenance and prevents invention or hallucination.
- **Rule**: Every fact, requirement, constraint, preference, or technology asserted in `PromptAnalysis` must trace directly back to verbatim authoritative user input (`<USER_REQUEST>` or `<USER_CLARIFICATIONS>`).
- **Defense**: Unprompted technologies, frameworks, and invented entities are flagged as `unsupported_items`, triggering analysis repair or pipeline halting.

### 2. SEMANTIC GROUNDING
> *"Did we interpret the user's words correctly?"*

- **Purpose**: Verifies that words present in the user request were assigned their correct contextual and syntactic role.
- **Rule**: Words cannot be co-opted for unrelated semantic roles merely because the token appeared in the text.
- **Examples**:
  - A user asking for a *"simple web app"* used "simple" as an artifact scope modifier; it must NOT be converted into an explanation style preference (*"simple explanation"*).
  - A user asking to design a *"professional portfolio"* used "professional" to describe an artifact; it must NOT be converted into a communication tone preference (*"professional tone"*).

### 3. CROSS-FIELD SEMANTIC COHERENCE
> *"Do all analysis fields describe the same requested task?"*

- **Purpose**: Enforces internal consistency across all derived analysis fields and locks classification to the user's requested action.
- **Rule**: All derived fields (`classification`, `objective`, `deliverable_semantics`, `explicit_requirements`, `constraints`, `preferences`, `missing_context`, `ambiguity`, and `selected_template`) must describe ONE coherent interpretation of the authoritative request.
- **Key Invariants**:
  - **Action Family, Not Topic Vocabulary**: Classification and intent anchoring follow *what the user is asking the AI to do* (action verbs and primary deliverable), never incidental subject matter nouns (e.g., creating a function that handles timestamps is `CODING`, not `DATA_ANALYSIS`; drafting a message regarding an invoice is `WRITING`, not `DEBUGGING`; conceptual difference questions are `EDUCATION`, not `GENERAL`).
  - **Strong vs. Weak Anchors**: Strong action anchors (`IntentAnchor`) deterministically constrain classification consistency. Weak anchors defer to analyzer semantic judgment.
  - **Task-Coherent Missing Context**: Cross-domain dimension leakage (such as injecting software framework dimensions into pure writing, education, or planning tasks) is rejected.
  - **Neutral Dimensions vs. Clarification Questions**: `missing_context` contains strictly neutral dimension names, never interrogative questions. User-facing questions remain segregated in `ambiguity.questions`.
  - **Self-Contained Operations**: Explicit transformations, calculations, comparisons, explanations, and well-defined drafting tasks already define their operational target; they are never falsely marked as lacking functional purpose.
  - **Validator Defense**: Detects `TASK_FAMILY_DRIFT` if the enhanced prompt converts to an incompatible task family, rejecting drift even if analysis and enhancement agree.

---

## Production Reliability & Boundary Invariants

### Core Invariant: INTERNAL CONTROL INSTRUCTIONS ARE NEVER USER SEMANTIC DATA

HumanIntent enforces a strict architectural boundary separating the **control-plane** (system prompts, numbered procedural steps, JSON schema contracts, evaluation rubrics, category templates) from the **data-plane** (authoritative user requests and clarification answers).

1. **Delimited Input Sources**: Authoritative user input is strictly encapsulated within structured boundary delimiters (`<USER_REQUEST>`, `<USER_CLARIFICATIONS>`). System instructions explicitly forbid extracting or echoing procedural control labels (`STEP 1`, `STEP 4`, `SYSTEM:`, `JSON SCHEMA`, etc.).
2. **Centralized Contamination Detection**: Every generated `PromptAnalysis` is exhaustively scanned across all fields (`objective`, `explicit_requirements`, `constraints`, `preferences`, `missing_context`, `ambiguity.reasons`, and `ambiguity.questions`) by `app.core.instruction_boundary`.
3. **Source-Based False-Positive Protection**: If a user genuinely requests procedural wording in their prompt (e.g., *"Provide Step 1 and Step 2 of the algorithm"*), the system recognizes its presence in authoritative input and permits it. Suspicious wording is rejected only when absent from the user's words.
4. **Controlled Repair & Fail-Safe Halting**: If instruction contamination or semantic incoherence is detected, the pipeline triggers at most ONE controlled analysis repair. If violations persist after repair, the pipeline halts immediately with `status: "analysis_failed"`, incurring 0 downstream model calls (no enhancer, validator, or scorer invocation).

---

## Phase 9 — Dataset Factory

The **HumanIntent Dataset Factory** (`dataset_factory/`) is an offline, production-grade infrastructure that produces clean, versioned, reproducible datasets for future model fine-tuning.

> **CRITICAL ARCHITECTURAL PRINCIPLE:**
> **Dataset Generation != Model Training.**
> Phase 9 builds data generation, curation, and export tooling. It does **not** train any model, execute fine-tuning, implement LoRA/QLoRA, or introduce vector databases.

### Factory Pipeline Flow

```text
Curated Seeds / Consented Real History
               │
               ▼
   Privacy & Credential Sanitizer (PII / Token Masking)
               │
               ▼
   HumanIntent Teacher Evaluation (Locked Phases 1–8 Pipeline)
               │
               ▼
   Staged Quality & Intent Filters (Grounding, Validation, Quality >= 75)
               │
               ▼
   Lexical Deduplication Engine (Exact Hash + Jaccard >= 0.85)
               │
               ▼
   Seed-Group Dataset Splitter (Zero Leakage: 90% Train / 10% Val)
               │
               ▼
   Versioned Dataset Exporter (SFT, Clarification, Preference JSONL + Manifest)
```

### Core Invariants:
1. **Offline Tooling**: Completely decoupled from the runtime API. Incurs 0 overhead on normal `/api/enhance` calls.
2. **Strict Consent Policy**: Only real records with `dataset_consent == True` are ever queried or loaded. Never infers consent from helpful feedback or user edits. Pseudonymizes real session identifiers.
3. **Privacy Sanitizer**: Redacts emails, phone numbers, API keys, bearer tokens, passwords, and private URLs. Unredactable cryptographic private keys trigger immediate candidate exclusion.
4. **Group-Based Splitting**: Train and validation splits are partitioned by Seed/Session group, guaranteeing that all variants of an intent remain in the same split to eliminate evaluation data leakage.
5. **Multi-Format Exports**: Emits SFT pairs (`sft_{train,val}.jsonl`), Clarification pairs (`clarification_{train,val}.jsonl`), Preference pairs (`preference_{train,val}.jsonl`), a Stratified Human Review Sample (`review_sample.jsonl`), build `manifest.json`, and an automated `DATASET_CARD.md`.



