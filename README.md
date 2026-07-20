# Bug Report Structurer

An adaptive pipeline that takes raw, unstructured bug reports and converts them into structured data — severity level, schema type, affected component, trigger action, and error signature — using a hybrid of a fine-tuned transformer model and an LLM.

## Overview

Bug trackers are full of inconsistently written reports: some are detailed, some are a single sentence, and severity/category labels are often missing or unreliable. This project builds a pipeline that automatically extracts structured fields from free-text bug descriptions, with a confidence-based triage system that flags uncertain predictions for human review instead of silently guessing.

The project evolved through several stages of experimentation before arriving at the current architecture:

1. **LLM prompting experiments** (zero-shot and few-shot, across model sizes) to establish a baseline for structured extraction from bug text.
2. **A fine-tuned DistilBERT classifier** for severity prediction, trained on labeled Bugzilla data, which outperformed all LLM prompting approaches.
3. **A hybrid inference pipeline** combining the fine-tuned model (severity) with LLM-based extraction (schema type, component, trigger action, error signature) — using each approach where it performs best.
4. **A FastAPI backend and lightweight frontend** to make the pipeline usable as a real service rather than a notebook experiment.

## Key finding: fine-tuning vs. prompting

| Approach | Accuracy | Macro F1 |
|---|---|---|
| Llama 8B, few-shot prompting | 39.3% | 33.1% |
| Llama 70B, few-shot prompting | 43.3% | 38.7% |
| **DistilBERT, fine-tuned (this project)** | **59.0%** | **54.8%** |

A 66M-parameter model fine-tuned for a few minutes on a free Colab GPU outperformed a 70B-parameter LLM using few-shot prompting on the same severity classification task. Full experiment details, confusion matrices, and per-class breakdowns are in `/reports`.

## Architecture

```
                 ┌─────────────────────┐
GitHub/Jira      │                     │      Web frontend
webhook  ───────▶│   FastAPI backend   │◀─────  (analyze, browse,
(planned)        │                     │         view stats)
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌────────────────────┐      ┌────────────────────┐
   │  Fine-tuned         │      │  LLM extraction     │
   │  DistilBERT         │      │  (Groq / Llama 70B) │
   │  → severity +       │      │  → schema_type,      │
   │    confidence       │      │    component,        │
   │                     │      │    trigger_action,   │
   │                     │      │    error_signature    │
   └─────────────────────┘      └────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   SQLite database    │
                 │   (persisted results)│
                 └─────────────────────┘
```

**Why hybrid?** Fine-tuning clearly outperformed prompting for severity classification, so that task runs on a local, fine-tuned model — fast, free, and more accurate. Schema type and free-text field extraction were already handled well by zero-shot LLM prompting (~80% accuracy in earlier experiments), so fine-tuning a separate model for that task wasn't necessary at this stage.

## Features

- **`POST /analyze`** — submit a raw bug description, get back a structured JSON object with severity, confidence score, schema type, component, trigger action, and error signature.
- **Confidence-based triage** — predictions below a configurable confidence threshold are flagged with `needs_human_review: true` instead of being silently trusted.
- **`GET /bugs`** — browse past analyses, filterable by severity, schema type, and review status.
- **`DELETE /bugs/{id}`** — remove a stored record.
- **`GET /stats`** — aggregate view: total bugs analyzed, severity/schema type distribution, percentage flagged for review, average confidence.
- **`GET /dashboard`** — a single-page web UI covering all of the above (analyze form, browsable history with filters, stats view).

## Example

Request:
```json
POST /analyze
{
  "description": "The application crashes immediately after opening the Settings page. This happens every time."
}
```

Response:
```json
{
  "schema_type": "backend",
  "component": "settings page",
  "trigger_action": "opening the settings page",
  "error_signature": "application crash",
  "severity": "critical",
  "severity_confidence": 0.87,
  "needs_human_review": false
}
```

## Confidence-based triage

Rather than trusting every prediction equally, the pipeline exposes the model's confidence and uses it to decide what gets automated vs. routed to a human:

| Confidence threshold | Auto-decided | Accuracy (auto) | Sent to review |
|---|---|---|---|
| 0.5 | 86.9% | 63.1% | 13.1% |
| 0.7 | 62.1% | 72.4% | 37.9% |
| 0.9 | 37.9% | 83.9% | 62.1% |

Predictions the model is unsure about are meaningfully less reliable (accuracy drops to ~32% in the lowest-confidence group at threshold 0.5), confirming that confidence is a useful signal for deciding which classifications need human oversight rather than being trusted outright.

## Tech stack

- **Model training**: PyTorch, Hugging Face Transformers, `distilbert-base-uncased`, trained on Google Colab (free tier, T4 GPU)
- **LLM extraction**: Groq API (Llama 3.3 70B)
- **Backend**: FastAPI, SQLite
- **Frontend**: single-page HTML/CSS/vanilla JS

## Project structure

```
bug_structure/
├── api/
│   ├── main.py            # FastAPI app, route definitions
│   ├── inference.py       # severity model + LLM extraction logic
│   └── database.py        # SQLite persistence layer
├── frontend/
│   └── index.html         # single-page dashboard (analyze / history / stats)
├── models/
│   └── severity_distilbert_4class/   # fine-tuned model weights
├── training/
│   └── train_severity_classifier.ipynb
├── data/
│   └── bugzilla_clean.csv
├── reports/
│   └── ...                # experiment writeups, confusion matrices
└── README.md
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install fastapi uvicorn transformers torch groq python-dotenv
   ```

2. Add a `.env` file inside `api/` with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```

3. Run the backend:
   ```bash
   cd api
   uvicorn main:app --reload
   ```

4. Open the dashboard:
   ```
   http://localhost:8000/dashboard
   ```

   Or explore the API directly via the auto-generated docs:
   ```
   http://localhost:8000/docs
   ```

## Roadmap

- [ ] GitHub/Jira webhook integration for automatic triage on new issues
- [ ] Write predictions back to the source tracker as labels
- [ ] Fine-tuned model for schema type classification (currently LLM-only, pending more labeled data)
- [ ] Docker containerization for easier deployment
- [ ] Class-weighted training to improve the weakest severity class (`medium`)

## Background

This project started as a research exercise comparing LLM prompting strategies (zero-shot vs. few-shot, model size, prompt structure) for structuring unstructured bug reports, then extended into fine-tuning and a working end-to-end service once the research findings pointed toward a hybrid approach as the strongest option.
