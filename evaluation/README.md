# HumanIntent Holdout Benchmark Infrastructure

This directory contains evaluation tooling to assess HumanIntent generalization against external, unseen holdout datasets.

## Purpose

To prevent overfitting to regression prompts and training examples during development:
- **No benchmark prompts** are stored in this repository.
- Tests against unseen prompts are executed using external JSONL files passed at runtime.

## Usage

### 1. In-process Evaluation (Direct Python)

Run directly using backend modules with your local Ollama instance:

```bash
python evaluation/run_holdout.py --file /path/to/external_holdout.jsonl
```

### 2. HTTP Evaluation (Running Backend Service)

Run against a live FastAPI server (e.g. `http://localhost:8000/api/enhance`):

```bash
python evaluation/run_holdout.py --file /path/to/external_holdout.jsonl --api-url http://localhost:8000/api/enhance
```

### 3. Saving Detailed Results

Optionally save output JSONL with full structured analyses, enhanced prompts, and invariant evaluations:

```bash
python evaluation/run_holdout.py --file /path/to/external_holdout.jsonl --output results.jsonl
```

## Input File Format (JSONL)

Each line in the external `.jsonl` file represents one test case:

```json
{"prompt": "A natural language request from a user", "expected_category": "CODING", "must_contain": ["Vue"], "must_not_contain": ["Bootstrap"]}
```

### Fields:
- `prompt` (*string, required*): The raw user prompt to analyze and enhance.
- `expected_category` (*string, optional*): The expected primary `PromptCategory` (e.g., `CODING`, `DEBUGGING`, `RESEARCH`, `EDUCATION`, `WRITING`, `DATA_ANALYSIS`, `PLANNING`, `CREATIVE`, `IMAGE_GENERATION`, `SUMMARIZATION`).
- `must_contain` (*list of strings, optional*): Terms or deliverables that must be preserved in the analysis or enhanced output.
- `must_not_contain` (*list of strings, optional*): Unprompted technologies, discarded alternatives, or forbidden candidate features that must NOT appear.

## Output Metrics Reported

For each prompt, `run_holdout.py` reports:
1. Classification category and confidence score.
2. Extracted objective.
3. Count and list of explicit requirements.
4. Count and list of constraints.
5. Count and list of preferences.
6. Count and list of missing context dimensions.
7. Length of enhanced output prompt.
8. Invariant violation check results.
