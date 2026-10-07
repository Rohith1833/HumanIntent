# HumanIntent
## Intent-Preserving AI Prompt Enhancement System

**Version:** 1.0  
**Project Type:** AI / NLP / Local LLM / Full-Stack Application  
**Primary LLM Runtime:** Ollama  
**Primary Model Family:** Meta Llama  
**Development Approach:** Local-first, free-first, modular, eventually fine-tunable

---

# 1. Product Summary

HumanIntent is an AI-powered prompt enhancement system that converts informal, incomplete, ambiguous, or poorly structured human requests into high-quality prompts that can be given to AI systems.

The system must not simply make prompts longer.

Its primary responsibility is to understand:

- what the user actually wants
- what information the user explicitly provided
- what information is missing
- what assumptions are safe
- what assumptions must not be made
- which AI/task type the prompt is intended for
- how much enhancement is necessary
- whether clarification is required

It then produces an improved prompt while preserving the user's original intention.

---

# 2. Problem Statement

Most users communicate with AI naturally.

Examples:

> make me portfolio website dark theme student

> explain binary search easy

> build login system python

> research ai healthcare

These requests are understandable to humans but often lack important details required for high-quality AI output.

Existing prompt enhancers frequently solve this by simply expanding the input into a large generic prompt.

This creates several problems:

- unnecessary prompt length
- invented requirements
- changed user intent
- excessive assumptions
- generic role prompting
- irrelevant constraints
- poor adaptation to different tasks
- no measurement of enhancement quality

HumanIntent aims to solve these problems.

---

# 3. Product Vision

Create a system that acts as a translator between:

**how humans naturally express ideas**

and

**how AI systems benefit from receiving instructions.**

Core philosophy:

Human Thought  
↓  
Understand  
↓  
Extract Intent  
↓  
Detect Missing Context  
↓  
Clarify When Necessary  
↓  
Structure  
↓  
Enhance  
↓  
Validate  
↓  
Score  
↓  
Optimized Prompt

---

# 4. Main Product Principle

## Intent Preservation

The enhanced prompt must preserve the meaning of the original request.

Example:

Input:

> make portfolio

The system must NOT automatically assume:

- React
- Next.js
- Tailwind CSS
- PostgreSQL
- Firebase
- AWS
- TypeScript

unless the user supplied those requirements or selected a mode that explicitly allows reasonable assumptions.

The system should instead create something like:

> Create a professional personal portfolio website. Focus on presenting information clearly and use placeholders for details or technologies that have not been provided.

Intent preservation is one of the core differentiators of the product.

---

# 5. Target Users

HumanIntent can serve:

- students
- developers
- researchers
- writers
- designers
- content creators
- AI beginners
- AI power users
- software teams
- people using local AI models

The first version will target general AI users.

---

# 6. Primary Use Cases

HumanIntent should support at least the following categories.

## Coding

Example:

> make attendance app python

## Debugging

Example:

> this api isn't saving my users fix it

## Research

Example:

> research blockchain sustainable finance

## Education

Example:

> explain os deadlock easy with example

## Writing

Example:

> write mail teacher absent yesterday

## Data Analysis

Example:

> analyze this csv and tell important patterns

## Planning

Example:

> give me roadmap learn machine learning

## Creative Tasks

Example:

> story about future city where ai controls traffic

## Image Prompting

Example:

> futuristic indian city at night

## General AI Tasks

Any request that does not fit the specialized categories.

---

# 7. Core User Flow

User enters:

> make me an ai resume analyzer using python

System performs:

### Stage 1 — Input Analysis

Detect:

Category:
Coding / AI application

Intent:
Build an AI resume analysis application

Explicit requirements:
- Python
- AI
- Resume analysis

Missing information:
- UI type
- file format
- LLM choice
- output structure

---

### Stage 2 — Ambiguity Analysis

System calculates how much important information is missing.

Example:

Ambiguity Score:

35 / 100

Low score:
Enhance automatically.

Medium score:
Enhance with clearly marked assumptions.

High score:
Ask a small number of clarification questions.

---

### Stage 3 — Enhancement

Create a structured prompt.

Example structure:

ROLE

OBJECTIVE

CONTEXT

FUNCTIONAL REQUIREMENTS

TECHNICAL REQUIREMENTS

CONSTRAINTS

EDGE CASES

OUTPUT FORMAT

ACCEPTANCE CRITERIA

---

### Stage 4 — Validation

Compare:

Original Prompt

vs.

Enhanced Prompt

Check whether:

- intent was preserved
- explicit requirements were retained
- requirements were not contradicted
- unnecessary assumptions were introduced
- unrelated instructions were added

---

### Stage 5 — Scoring

Display quality information.

Example:

Intent Preservation: 97

Clarity: 94

Context: 78

Specificity: 86

Constraint Preservation: 100

Overall Prompt Quality: 91

---

### Stage 6 — User Action

User can:

- copy prompt
- regenerate prompt
- edit prompt
- change enhancement level
- answer clarification questions
- thumbs up
- thumbs down

Later versions can also store approved edits for dataset improvement.

---

# 8. Enhancement Modes

The product will contain multiple enhancement levels.

## Mode 1 — Quick

Goal:

Clean the prompt without making it much longer.

Input:

> explain binary search easy

Output:

> Explain binary search in simple beginner-friendly language. Include a small example and explain how the algorithm finds an element step by step.

---

## Mode 2 — Professional

Adds:

- context
- requirements
- structure
- constraints
- output expectations

This will be the default mode.

---

## Mode 3 — Deep

Used for complex tasks.

Adds:

- architecture expectations
- edge cases
- assumptions
- acceptance criteria
- detailed output structure
- validation expectations

---

## Mode 4 — Agent

Designed for coding/research agents.

Adds instructions such as:

- inspect existing context
- create an implementation plan
- avoid unnecessary modifications
- verify changes
- handle errors
- follow project structure
- complete work systematically

This mode should be added after the base enhancer works.

---

# 9. Prompt Classification Engine

Every prompt will first be classified.

Initial categories:

CODING

DEBUGGING

RESEARCH

EDUCATION

WRITING

DATA_ANALYSIS

PLANNING

CREATIVE

IMAGE_GENERATION

SUMMARIZATION

GENERAL

The classifier returns structured JSON.

Example:

```json
{
  "category": "CODING",
  "confidence": 0.93
}
```

---

# 10. Intent Extraction Engine

The system extracts structured meaning from the user's input.

Example:

Input:

> make python attendance system for college

Output:

```json
{
  "intent": "build attendance management system",
  "domain": "software development",
  "explicit_requirements": [
    "Python",
    "college attendance"
  ],
  "constraints": [],
  "preferences": [],
  "missing_context": [
    "application type",
    "database",
    "authentication"
  ]
}
```

This internal representation is extremely important.

The final prompt should not be generated directly from raw text.

It should be generated from the extracted intent representation.

---

# 11. Missing Context Detector

The system determines what useful information was not provided.

Example:

User:

> make ecommerce website

Possible missing information:

- frontend technology
- backend requirement
- database requirement
- authentication
- payment requirement
- deployment target

Not every missing field must be asked.

The system determines which missing information significantly changes the solution.

---

# 12. Clarification Engine

The enhancer should avoid interrogating the user.

Maximum recommended clarification questions:

3.

Questions must be high-impact.

Example:

Instead of asking:

- color?
- font?
- database?
- language?
- frontend?
- backend?
- hosting?
- animation?
- authentication?

Ask:

1. Should this be a web, desktop, or mobile application?
2. Do you already have a preferred technology stack?
3. Should users have accounts/login functionality?

The user should also have:

**Use reasonable assumptions**

as an option.

---

# 13. Ambiguity Engine

An ambiguity score estimates how incomplete the request is.

Initial conceptual formula:

Ambiguity Score =
Missing Critical Information
+
Unclear Objective
+
Conflicting Requirements
+
Undefined Output

Example:

> make app

Ambiguity:

95 / 100

Example:

> Build a Python FastAPI REST API using PostgreSQL and JWT authentication for a college attendance application.

Ambiguity:

15 / 100

Initially this score can be produced using a combination of:

- deterministic rules
- LLM analysis

Later it can become its own ML problem.

---

# 14. Prompt Template Engine

Different tasks need different enhancement strategies.

The system must NOT use one universal template.

---

## Coding Template

ROLE

OBJECTIVE

PROJECT CONTEXT

FUNCTIONAL REQUIREMENTS

TECH STACK

TECHNICAL REQUIREMENTS

CONSTRAINTS

EDGE CASES

OUTPUT FORMAT

ACCEPTANCE CRITERIA

---

## Research Template

ROLE

RESEARCH QUESTION

OBJECTIVE

BACKGROUND

SCOPE

METHODOLOGY

EVIDENCE REQUIREMENTS

ANALYSIS

LIMITATIONS

OUTPUT STRUCTURE

---

## Education Template

TOPIC

STUDENT LEVEL

LEARNING OBJECTIVE

EXPLANATION STYLE

REQUIRED EXAMPLES

DIFFICULTY

OUTPUT FORMAT

---

## Writing Template

PURPOSE

AUDIENCE

TONE

KEY INFORMATION

CONSTRAINTS

LENGTH

OUTPUT FORMAT

---

## Data Analysis Template

OBJECTIVE

DATA CONTEXT

QUESTIONS TO ANSWER

ANALYSIS REQUIREMENTS

STATISTICAL REQUIREMENTS

VISUALIZATION REQUIREMENTS

OUTPUT STRUCTURE

---

## Image Prompt Template

SUBJECT

ENVIRONMENT

COMPOSITION

LIGHTING

CAMERA

VISUAL STYLE

DETAILS

COLOR

CONSTRAINTS

---

# 15. Prompt Generator

The Prompt Generator receives:

- original input
- classification
- intent
- requirements
- missing context
- enhancement mode
- target model/type

It then produces the enhanced prompt.

Architecture:

Raw Prompt

↓

Intent Analyzer

↓

Structured Intent JSON

↓

Template Selector

↓

Prompt Generator

↓

Enhanced Prompt

This reduces hallucinated requirements compared with directly asking:

> Make this prompt better.

---

# 16. Prompt Validator

After generation, another evaluation step checks the output.

Validator input:

Original prompt

+

Extracted intent

+

Enhanced prompt

Validator output:

```json
{
  "intent_preserved": true,
  "requirements_preserved": true,
  "unsupported_assumptions": [],
  "contradictions": [],
  "validation_score": 0.94
}
```

If validation fails:

Enhanced Prompt

↓

Validator

↓

FAIL

↓

Regenerate

This creates a self-correction loop.

Maximum regeneration attempts should initially be limited to approximately 2.

---

# 17. Prompt Quality Scoring

Initial score dimensions:

Intent Clarity

Context Quality

Requirement Coverage

Constraint Preservation

Output Definition

Specificity

Intent Preservation

Unnecessary Assumptions

A possible initial weighting:

Intent Preservation: 25%

Clarity: 15%

Requirement Coverage: 15%

Context: 10%

Constraints: 10%

Output Definition: 10%

Specificity: 10%

Structure: 5%

This is not intended to represent absolute scientific truth.

It is an explainable product metric that can later be improved through evaluation data.

---

# 18. Local AI Architecture

HumanIntent will be local-first.

Main stack:

Frontend

React / Next.js

Backend

Python

FastAPI

AI Runtime

Ollama

LLM

Llama

Database

SQLite initially

PostgreSQL later if required

Embeddings

Dedicated local embedding model through Ollama

ML libraries later:

PyTorch

Transformers

Datasets

PEFT

TRL

---

# 19. Ollama Architecture

The application does not run the Llama model directly.

Architecture:

HumanIntent Backend

↓

Ollama HTTP API

↓

Llama Model

↓

Generated Response

Example development model:

llama3.1:8b

Fallback development model:

llama3.2:3b

The model name must be configurable through environment settings.

Example:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

Never hard-code one model throughout the application.

---

# 20. Why Ollama

Ollama gives the project:

- local inference
- no paid API requirement
- model switching
- privacy advantages
- offline development capability
- access through a local API
- simple experimentation with different models

The product architecture should still contain an abstraction layer so a future AI provider could replace Ollama without rewriting the entire application.

Example:

```text
LLMProvider
    |
    +-- OllamaProvider
    |
    +-- FutureProvider
```

For Version 1:

Only OllamaProvider needs to exist.

---

# 21. LLM Service Layer

Backend structure:

```text
llm/
├── base.py
├── ollama_client.py
├── schemas.py
└── prompts/
```

Functions might eventually include:

```python
classify_prompt()

extract_intent()

detect_missing_context()

enhance_prompt()

validate_prompt()

score_prompt()
```

Initially multiple operations can use the same Llama model.

Later different specialized models can handle individual stages.

---

# 22. Structured LLM Responses

Where possible, internal model operations should return JSON.

Example classifier:

```json
{
  "category": "CODING",
  "confidence": 0.92
}
```

Not:

> I believe this is probably a programming-related question.

Structured responses make backend programming significantly easier.

---

# 23. Dataset Strategy

Training a model is NOT Phase 1.

First:

Build the product.

Then:

Collect and generate data.

Dataset pipeline:

Seed Intents

↓

Synthetic Variations

↓

Messy Human Prompts

↓

Intent Extraction

↓

Teacher Enhancement

↓

Multiple Candidates

↓

Quality Evaluation

↓

Deduplication

↓

Human Review

↓

Dataset

---

# 24. Dataset Example

```json
{
  "id": "coding_00001",
  "original": "make python login api fastapi postgres",
  "category": "CODING",
  "intent": "build authentication API",
  "explicit_requirements": [
    "Python",
    "FastAPI",
    "PostgreSQL"
  ],
  "missing_context": [
    "authentication method"
  ],
  "enhancement_level": "professional",
  "enhanced": "Act as an experienced backend developer..."
}
```

---

# 25. Dataset Generation Strategy

Start small.

Stage 1:

100 manually designed seed intents.

Stage 2:

Generate approximately 5 realistic human variants each.

100 × 5

=

500 examples.

Stage 3:

Run automatic enhancement.

Stage 4:

Generate multiple candidate enhancements.

Stage 5:

Automatically evaluate candidates.

Stage 6:

Human-review the strongest and weakest samples.

Stage 7:

Create the first clean dataset.

Then scale:

500

↓

2,000

↓

5,000

↓

10,000+

Quality matters more than raw size.

---

# 26. Prompt Messifier

One component intentionally converts clean instructions into realistic human language.

Clean:

> Create a responsive React portfolio website for a computer science student.

Variants:

> make react portfolio for cs student responsive

> need portfolio website react student something modern

> build me responsive student portfolio react

> react portfolio. student. responsive.

Different messiness levels:

Level 0:
Already high quality

Level 1:
Minor grammar problems

Level 2:
Casual request

Level 3:
Incomplete request

Level 4:
Highly ambiguous request

This prevents the final model from only understanding perfectly written inputs.

---

# 27. Synthetic Data Validation

Generated examples must pass automatic filters.

Reject examples if:

- intent changed
- original constraints disappeared
- technologies were unnecessarily invented
- enhanced output contains contradictions
- model answered the task instead of creating a prompt
- JSON is invalid
- prompt contains excessive repetition
- enhanced prompt is unnecessarily huge
- example is a semantic duplicate

---

# 28. Deduplication

Dataset diversity is important.

Example:

These are nearly identical:

> make portfolio

> build portfolio

> create portfolio

> need portfolio

We will later generate embeddings for samples.

Process:

Prompt

↓

Embedding Model

↓

Vector

↓

Similarity Comparison

↓

Potential Duplicate

Possible implementation:

sentence-transformers

or

Ollama embedding model

Potential threshold:

Cosine similarity > 0.92

Flag for duplicate review.

The exact threshold will be experimentally determined.

---

# 29. Preference Dataset

Later we should create:

```json
{
  "prompt": "make python login backend",
  "chosen": "high quality enhancement",
  "rejected": "bad enhancement with invented technologies"
}
```

This teaches a later model what not to do.

---

# 30. Real User Feedback Dataset

When users interact with the application:

Generated Prompt A

↓

User edits it

↓

Prompt B

With proper user consent, anonymization, and privacy controls:

Prompt B can represent a human-preferred version.

Possible record:

```json
{
  "original": "...",
  "generated": "...",
  "edited": "...",
  "accepted": true
}
```

This becomes much more valuable than unlimited synthetic generation.

---

# 31. Fine-Tuning Phase

Fine-tuning is a later milestone.

Potential strategy:

Base Model

+

Clean Prompt Dataset

↓

Supervised Fine-Tuning

↓

LoRA / QLoRA Adapter

↓

Prompt-Specialized Model

The first model does not need to replace the complete pipeline.

Possible future architecture:

Fine-Tuned Prompt Model

↓

Validator

↓

Scorer

↓

Final Enhanced Prompt

---

# 32. Frontend Requirements

Main interface:

```text
┌─────────────────────────────────────────────┐
│               HumanIntent                   │
│                                             │
│ Turn rough ideas into powerful AI prompts   │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Describe what you want...               │ │
│ │                                         │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Enhancement                                │
│ [Quick] [Professional] [Deep] [Agent]      │
│                                             │
│ Target                                     │
│ [General AI ▼]                             │
│                                             │
│            [ Enhance Prompt ]              │
└─────────────────────────────────────────────┘
```

After enhancement:

```text
Detected

Coding
Confidence: 94%

Intent
Build an AI resume analyzer

Ambiguity
32%

────────────────────────

Enhanced Prompt

ROLE
...

OBJECTIVE
...

REQUIREMENTS
...

────────────────────────

Quality Score: 91

[Copy]

[Regenerate]

[Edit]

👍  👎
```

---

# 33. UI Philosophy

The UI should be:

- simple
- fast
- minimal
- focused
- modern
- responsive

Avoid:

- unnecessary dashboards
- excessive animations
- dozens of controls
- overwhelming settings

The prompt box and output are the central product.

---

# 34. Backend API

Initial endpoints:

```text
POST /api/analyze
POST /api/enhance
POST /api/validate
POST /api/feedback
GET  /api/health
GET  /api/models
```

Eventually we might combine several operations behind:

```text
POST /api/enhance
```

Internally that endpoint runs the full pipeline.

---

# 35. Example API Input

```json
{
  "prompt": "make attendance app python",
  "mode": "professional",
  "target": "general"
}
```

---

# 36. Example API Response

```json
{
  "original_prompt": "make attendance app python",

  "analysis": {
    "category": "CODING",
    "intent": "build attendance application",
    "confidence": 0.95,
    "ambiguity": 0.42
  },

  "enhanced_prompt": "...",

  "scores": {
    "clarity": 92,
    "intent_preservation": 97,
    "specificity": 84,
    "overall": 91
  }
}
```

---

# 37. Database

Version 1:

SQLite.

Tables:

## prompts

```text
id

original_prompt

enhanced_prompt

category

mode

target

created_at
```

## analysis

```text
prompt_id

intent

requirements

missing_context

ambiguity

confidence
```

## scores

```text
prompt_id

clarity

intent_preservation

specificity

overall
```

## feedback

```text
prompt_id

rating

edited_prompt

created_at
```

---

# 38. Privacy

Prompt data can contain sensitive information.

Therefore:

- do not silently collect prompts for training
- dataset contribution should be optional
- avoid storing unnecessary personal information
- provide deletion controls where user accounts/data storage exist
- sanitize obvious secrets before dataset inclusion
- never place passwords or API keys into datasets
- keep local mode available

This is also an advantage of Ollama.

---

# 39. Project Repository Structure

```text
humanintent/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── services/
│   └── types/
│
├── backend/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── analyze.py
│   │   ├── enhance.py
│   │   ├── feedback.py
│   │   └── health.py
│   │
│   ├── core/
│   │   ├── classifier.py
│   │   ├── intent.py
│   │   ├── ambiguity.py
│   │   ├── clarification.py
│   │   ├── enhancer.py
│   │   ├── validator.py
│   │   └── scorer.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── ollama_client.py
│   │   └── schemas.py
│   │
│   ├── templates/
│   │   ├── coding.py
│   │   ├── research.py
│   │   ├── education.py
│   │   ├── writing.py
│   │   ├── data_analysis.py
│   │   ├── image.py
│   │   └── general.py
│   │
│   ├── database/
│   │   ├── database.py
│   │   ├── models.py
│   │   └── schemas.py
│   │
│   └── evaluation/
│       ├── quality.py
│       └── metrics.py
│
├── dataset_factory/
│   ├── generation/
│   │   ├── seed_generator.py
│   │   ├── messifier.py
│   │   ├── candidate_generator.py
│   │   └── teacher.py
│   │
│   ├── cleaning/
│   │   ├── deduplicate.py
│   │   ├── filters.py
│   │   └── balance.py
│   │
│   ├── evaluation/
│   │   ├── intent_check.py
│   │   ├── constraint_check.py
│   │   └── quality_judge.py
│   │
│   └── output/
│
├── datasets/
│   ├── raw/
│   ├── cleaned/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── training/
│   ├── prepare.py
│   ├── train_lora.py
│   ├── evaluate.py
│   └── export.py
│
├── tests/
│
├── .env.example
├── README.md
└── requirements.txt
```

---

# 40. Product Architecture

```text
                    USER
                     │
                     ▼
                FRONTEND
                     │
                     ▼
              FastAPI Backend
                     │
                     ▼
             Prompt Analyzer
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     Classifier    Intent    Missing Info
          │          │          │
          └──────────┼──────────┘
                     ▼
               Ambiguity Engine
                     │
              ┌──────┴──────┐
              ▼             ▼
           Low/Med         High
              │             │
              │        Clarification
              │             │
              └──────┬──────┘
                     ▼
              Template Selector
                     │
                     ▼
                Prompt Builder
                     │
                     ▼
                  Ollama
                     │
                     ▼
               Llama Model
                     │
                     ▼
             Enhanced Prompt
                     │
                     ▼
                Validator
                     │
                     ▼
                  Scorer
                     │
                     ▼
                 FRONTEND
```

---

# 41. Important Architecture Decision

We will NOT start with six separate LLM calls for every prompt.

That could become unnecessarily slow locally.

Initially we will experiment with one analysis request that returns:

```json
{
  "category": "...",
  "intent": "...",
  "requirements": [],
  "missing_context": [],
  "ambiguity": 0.3
}
```

Then another request generates the enhanced prompt.

Then potentially a third request performs validation/scoring.

Initial architecture:

Input

↓

Analysis Call

↓

Enhancement Call

↓

Validation Call

Three major LLM operations.

Later we benchmark whether some stages can be merged.

---

# 42. Performance Goal

For local development:

The product should feel responsive.

We should measure:

- model loading time
- analysis latency
- generation latency
- validation latency
- total request time
- token usage
- model memory usage

Performance optimization happens after correctness.

---

# 43. Evaluation Dataset

We need a small manually verified benchmark that is never used for training.

Example:

```text
evaluation/
└── benchmark.jsonl
```

Categories should include:

- already-good prompts
- extremely vague prompts
- coding prompts
- research prompts
- educational prompts
- conflicting prompts
- prompts containing explicit constraints
- prompts where assumptions must not be added

This benchmark tells us whether changes actually improve the system.

---

# 44. Key Evaluation Metrics

We care about:

## Intent Preservation

Did enhancement change what the user wanted?

## Requirement Recall

Were explicit requirements retained?

## Hallucinated Requirement Rate

How many unsupported requirements were introduced?

## Prompt Quality

Does the enhanced prompt provide better instructions?

## Prompt Inflation

How much unnecessary text was added?

## Clarification Accuracy

Did the system ask questions only when necessary?

## Human Preference

Which version do users prefer?

These metrics make the project much stronger than:

> We asked the model and it looked good.

---

# 45. MVP Scope

Version 0.1 should contain ONLY:

- text input
- Ollama connection
- Llama generation
- prompt classification
- intent extraction
- one enhancement mode
- enhanced prompt output
- copy button

NO:

- model training
- accounts
- complex dashboard
- huge dataset
- analytics platform
- agent mode
- cloud deployment
- preference training

We first prove the fundamental idea works.

---

# 46. Version 0.2

Add:

- Quick mode
- Professional mode
- Deep mode
- categories
- ambiguity score
- missing context
- validation
- quality scoring

---

# 47. Version 0.3

Add:

- clarification questions
- user editing
- thumbs up/down
- SQLite storage
- history
- prompt comparison

---

# 48. Version 0.4

Dataset Factory.

Add:

- seed prompts
- synthetic messifier
- candidate generation
- quality judging
- automatic filtering
- deduplication
- dataset export

---

# 49. Version 0.5

Evaluation System.

Add:

- benchmark dataset
- model comparison
- enhancement metrics
- regression testing
- category performance

---

# 50. Version 0.6

Fine-Tuning Experiment.

Add:

- cleaned dataset
- train/validation/test split
- LoRA/QLoRA experiment
- before/after comparison

---

# 51. Version 1.0

Complete HumanIntent release.

Potential features:

- intelligent prompt enhancement
- clarification engine
- intent preservation
- multiple enhancement modes
- local LLM
- quality scoring
- feedback
- dataset factory
- evaluation suite
- optional specialized model

---

# 52. Development Roadmap

We will build this in the following order.

## PHASE 0 — Environment

Learn:

- what Ollama is
- what an LLM is
- what inference means
- what an API is
- how FastAPI communicates with Ollama

Build:

```text
Python
  ↓
Ollama API
  ↓
Llama
  ↓
Response
```

Definition of Done:

Python can send a prompt to local Llama and print the result.

---

## PHASE 1 — Basic Prompt Enhancer

Learn:

- system prompts
- user prompts
- temperature
- model output
- prompt engineering

Build:

```text
messy prompt
    ↓
Llama
    ↓
enhanced prompt
```

No frontend yet.

Definition of Done:

Terminal enhancer works.

---

## PHASE 2 — Structured AI Output

Learn:

- JSON
- schemas
- structured generation
- why AI output needs validation

Build:

```text
prompt
 ↓
analysis
 ↓
{
 category,
 intent,
 requirements,
 missing_context
}
```

Definition of Done:

The analyzer reliably returns structured data.

---

## PHASE 3 — Intent Engine

Learn:

- intent classification
- information extraction
- explicit vs inferred requirements

Build:

Classifier

+

Intent Extractor

+

Requirement Extractor

Definition of Done:

Messy human text becomes structured intent.

---

## PHASE 4 — Enhancement Engine

Learn:

- conditional prompting
- prompt templates
- task-specific prompting

Build:

Template selector.

Coding requests receive coding templates.

Research requests receive research templates.

Etc.

Definition of Done:

Enhancement changes based on task category.

---

## PHASE 5 — Ambiguity + Clarification

Learn:

- uncertainty
- missing information
- importance ranking

Build:

Ambiguity score.

Clarification decision.

Definition of Done:

The system knows when not to blindly enhance.

---

## PHASE 6 — Validator

Learn:

- LLM-as-judge
- rule-based evaluation
- semantic consistency

Build:

Original

vs.

Enhanced

validation.

Definition of Done:

Unsupported assumptions can be detected.

---

## PHASE 7 — Scoring

Learn:

- metrics
- weighting
- evaluation design

Build:

Prompt quality score.

Definition of Done:

Before/after quality can be compared.

---

## PHASE 8 — FastAPI Backend

Learn:

- REST APIs
- requests
- responses
- async programming
- backend architecture

Build:

HumanIntent backend.

Definition of Done:

Frontend-independent API works.

---

## PHASE 9 — Frontend

Learn:

- React/Next.js communication
- API requests
- state
- UI components

Build:

Full web interface.

Definition of Done:

User can enhance prompts through the browser.

---

## PHASE 10 — Database + Feedback

Learn:

- persistence
- relational databases
- feedback data

Build:

SQLite database.

Definition of Done:

Prompt history and ratings can be stored.

---

## PHASE 11 — Dataset Factory

Learn:

- synthetic data
- data cleaning
- dataset bias
- train/test contamination
- deduplication

Build:

Automatic dataset pipeline.

Definition of Done:

One command can generate and clean dataset samples.

---

## PHASE 12 — Evaluation

Learn:

- benchmarking
- precision
- regression testing
- qualitative vs quantitative metrics

Build:

HumanIntent benchmark.

Definition of Done:

We can objectively compare versions.

---

## PHASE 13 — Fine-Tuning

Learn:

- tokens
- parameters
- transformers
- supervised fine-tuning
- LoRA
- QLoRA
- epochs
- learning rates
- overfitting
- validation loss

Build:

First specialized prompt enhancer model.

Definition of Done:

Fine-tuned model beats the original base model on our benchmark.

---

# 53. How We Will Learn While Building

For every phase we follow:

### 1. Concept

I explain what the technology/concept means.

### 2. Why

You understand why HumanIntent needs it.

### 3. Architecture

We see where it fits.

### 4. Small Example

We create a tiny standalone example.

### 5. Implementation

We put it inside HumanIntent.

### 6. Test

We test normal and weird inputs.

### 7. Debug

If something fails, we understand why instead of randomly changing code.

### 8. Review

You should be able to explain what we built in your own words.

Only then do we move to the next major component.

---

# 54. Rule for Using AI Coding Agents

AI can help us write code.

But we will not blindly tell an agent:

> build the entire project.

For every major module we first understand:

Input

Processing

Output

Dependencies

Failure cases

Then AI can assist with implementation.

That prevents ending up with a large codebase that works but that the developer cannot explain.

---

# 55. Free-First Requirement

The core project should remain possible without paid APIs.

Required:

- laptop
- Python
- Node.js
- Git
- Ollama
- local Llama model
- SQLite
- free/open-source libraries

Paid services are not required for core functionality.

---

# 56. Current Recommended Local Model Setup

Primary development model:

```bash
ollama run llama3.1:8b
```

If performance is poor:

```bash
ollama run llama3.2:3b
```

The final model choice will be decided using benchmarks rather than assumptions.

We can later compare:

Model A

vs.

Model B

using exactly the same HumanIntent test dataset.

---

# 57. Initial Success Criteria

The project is successful when:

A vague prompt such as:

> build attendance system python college

can be understood as:

- software development
- attendance management
- Python
- college context

and converted into a useful structured prompt without inventing unnecessary technologies.

At the same time, a detailed prompt should not become dramatically longer for no reason.

---

# 58. Final Success Criteria

HumanIntent should eventually demonstrate:

1. Better prompt clarity.

2. High intent preservation.

3. Low unsupported-assumption rate.

4. Multiple task categories.

5. Intelligent clarification.

6. Explainable quality scores.

7. Local inference through Ollama.

8. Synthetic dataset generation.

9. Human feedback collection.

10. A reproducible evaluation framework.

11. Fine-tuning experiments.

12. Measurable comparison against the base Llama model.

---

# 59. Final Product Positioning

HumanIntent should NOT be described simply as:

> A website that improves prompts.

It should be positioned as:

**HumanIntent — An Intent-Preserving Adaptive Prompt Optimization System**

Research/project description:

HumanIntent is a local-first NLP system designed to transform informal human requests into structured AI instructions while preserving original intent, minimizing unsupported assumptions, detecting ambiguity, requesting high-value clarification, and evaluating prompt quality through a multi-stage analysis and validation pipeline.

---

# 60. Ultimate Architecture

```text
                 HUMAN THOUGHT
                      │
                      ▼
                 RAW PROMPT
                      │
                      ▼
            ┌────────────────────┐
            │   HUMANINTENT      │
            ├────────────────────┤
            │ Classification     │
            │ Intent Extraction  │
            │ Requirement Mining │
            │ Missing Context    │
            │ Ambiguity          │
            │ Clarification      │
            │ Template Selection │
            │ Prompt Generation  │
            │ Validation         │
            │ Quality Scoring    │
            └──────────┬─────────┘
                       │
                       ▼
               OPTIMIZED PROMPT
                       │
                       ▼
                    ANY AI


                  FEEDBACK
                       │
                       ▼
               DATASET FACTORY
                       │
                       ▼
                 CLEAN DATASET
                       │
                       ▼
                  FINE-TUNING
                       │
                       ▼
              HUMANINTENT MODEL
```

---

# 61. Project Rule

We are not building the model first.

The order is:

**System → Data → Evaluation → Model**

Not:

**Model → hope it works.**

The application creates the environment where high-quality data can eventually be generated.

That data creates the possibility of training a specialized model.

---

# 62. First Development Milestone

The first milestone is intentionally tiny:

```text
Human enters:
"make python portfolio website"

        ↓

Python Backend

        ↓

Ollama

        ↓

Llama

        ↓

HumanIntent System Prompt

        ↓

Enhanced Prompt

        ↓

Terminal Output
```

Once this works reliably, we begin separating:

Intent Analysis

Classification

Missing Context

Prompt Generation

Validation

Scoring

That is where HumanIntent starts becoming an actual AI system rather than a basic LLM wrapper.
