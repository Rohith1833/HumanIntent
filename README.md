# HumanIntent

> **Intent-Preserving Adaptive Prompt Optimization System**

HumanIntent translates informal, incomplete, or ambiguous human requests into clear, high-quality, structured prompts for AI systems—**without silently inventing unrequested technologies, frameworks, or facts, and without making prompts arbitrarily long.**

---

## Current Status: Phase 7 — Explainable Prompt Quality Scoring

This repository implements **Phase 7**:
- **Explainable Prompt Quality Scoring**: After a prompt successfully passes intent-preservation validation, HumanIntent estimates how effective, well-formed, and clear the resulting prompt is.
- **Validation != Quality Scoring**:
  - *Validation* asks: *"Did we faithfully preserve the user's intent?"* (Pass/Fail with structured issues).
  - *Quality Scoring* asks: *"How effective and clear is the resulting prompt?"* (0–100 explainable dimensional breakdown).
  - Quality scoring runs **ONLY** after validation succeeds. If validation fails, the prompt is repaired; if repair fails, no quality score is calculated.
- **8 Explicit Scoring Dimensions**:
  1. **Intent Preservation (25%)**: Evaluated using validator evidence (100 for clean validation, 95 for info notes, 85 for advisory warnings).
  2. **Requirement Coverage (20%)**: Deterministic token/stemming analysis comparing `analysis.explicit_requirements` to the enhanced prompt.
  3. **Constraint Preservation (15%)**: Strict verification that negative restrictions and quantitative limits are retained.
  4. **Clarity (15%)**: Rubric-based semantic score for direct, unambiguous instructions.
  5. **Specificity (10%)**: Rewarding useful concrete detail without inventing unsupported facts (longer prompts do not automatically score higher).
  6. **Output Definition (7%)**: Clear communication of expected deliverable format and structure.
  7. **Structure (5%)**: Evaluated relative to task complexity (simple requests can score 100 with 2–3 lines and zero headings; complex requests benefit from sections).
  8. **Concision (3%)**: Signal-to-noise ratio and absence of boilerplate padding.
- **Deterministic Overall Aggregation**: The overall score is strictly computed via centralized fixed weights (`overall_score = round(sum(w_i * s_i))`). The LLM does NOT produce the overall score.
- **Prompt Inflation Diagnostics**: Relative expansion ratio (`enhanced_words / original_words`) and categorical risk (`LOW`, `MEDIUM`, `HIGH`) considering initial prompt complexity.
- **Strict Call Budget**:
  - Normal clear request: 1 analysis + 1 enhancement + 1 validation + 1 scoring = **4 LLM calls**.
  - Repaired request: 1 analysis + 2 enhancement + 2 validation + 1 scoring = **6 LLM calls max**.
  - Ambiguous request: 1 analysis + 0 enhancement + 0 validation + 0 scoring = **1 LLM call**.

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
uvicorn app.main:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`
- Analyze endpoint: `POST http://localhost:8000/api/analyze`
- Enhance endpoint: `POST http://localhost:8000/api/enhance`

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
The model is fully environment-driven and configurable. Any model pulled in Ollama can be used by setting `OLLAMA_MODEL`:
```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=120
BACKEND_CORS_ORIGINS=["http://localhost:3000"]
```

### Frontend (`frontend/.env.local`)
```ini
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## Running Tests

### Backend Unit & Regression Tests (134 Tests)
Automated tests mock the Ollama client layer, running in < 3s without requiring a live Ollama instance:
```powershell
cd backend
python -m pytest tests -v
```

### Frontend Build Validation
```powershell
cd frontend
npm run build
```
