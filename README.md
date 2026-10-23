# Bug Report Structurer 🐞

Turns raw, free-text software bug reports into structured data. A fine-tuned DistilBERT model predicts severity locally, an LLM on Groq extracts the technical fields, and every result is stored in a local SQLite database and shown in a web dashboard.

## How it works

| Part | What it does |
|---|---|
| **Local DistilBERT (fine-tuned)** | Classifies `severity` (`low`, `medium`, `high`, `critical`) and returns a confidence score. Results below **0.7** confidence are flagged `needs_human_review`. |
| **Groq API (`openai/gpt-oss-20b`)** | Extracts `schema_type` (`backend`, `frontend`, `database`, `performance`, `other`), `component`, `trigger_action` and `error_signature` as JSON. |
| **FastAPI** | Serves the API and the dashboard, and stores every analyzed report in SQLite (`api/bugs.db`, created automatically on first start). |

Example response from `POST /analyze`:

```json
{
  "schema_type": "backend",
  "component": "auth_service",
  "trigger_action": "token refresh",
  "error_signature": "Can't locate object method 'realname'",
  "severity": "critical",
  "severity_confidence": 0.91,
  "needs_human_review": false
}
```

---

## 🚀 Setup Guide

**Requirements:** Python 3.10 or newer, and a free [Groq API key](https://console.groq.com/keys).

### 1. Clone the repository

```bash
git clone https://github.com/alperenakksoy/bug_structure.git
cd bug_structure
```

### 2. Download the local AI model (crucial step)

To keep the GitHub repository lightweight, the model weights (`model.safetensors`) are hosted externally. The server will not start without this file.

**This is the only file you need to download.** All other model files (`config.json`, tokenizer files, etc.) are already in the repository.

1. Download it here: [Download model.safetensors](https://drive.google.com/file/d/1Q6jheVj332YWVLrwisqX4o47xAGyX4H5/view?usp=drive_link)
2. Place it in `models/severity_distilbert_4class/`, next to the existing files.

### 3. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch and Transformers, so the first install can take a few minutes.

### 5. Configure your API key

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_actual_api_key_here
```

### 6. Run the server

Run this from the project root:

```bash
uvicorn api.main:app --reload --reload-exclude ".venv"
```

### 7. Open the dashboard

Go to <http://localhost:8000/dashboard>. The dashboard has tabs to:

- **Analyze** a bug report: paste any raw bug description and click **Analyze**.
- **Browse** previously analyzed reports, with filters and delete.
- **View stats**: severity and schema-type distribution, share needing review, average confidence.

> `http://localhost:8000/` alone shows "Not Found". The dashboard lives at `/dashboard`.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Body: `{"description": "..."}`. Analyzes and stores a bug report. |
| `GET` | `/bugs` | Lists stored reports. Optional filters: `severity`, `schema_type`, `needs_review`, `limit`. |
| `DELETE` | `/bugs/{id}` | Deletes a stored report. |
| `GET` | `/stats` | Aggregate statistics. |
| `GET` | `/health` | Health check. |
| `GET` | `/dashboard` | Web dashboard. |

Interactive API docs are available at <http://localhost:8000/docs>.

---

## Evaluation (optional)

The `evaluation/` and `pipeline/` folders compare LLM-only extraction (Llama 3.1 8B, Llama 3.3 70B, Qwen; zero-shot and few-shot) on the cleaned Bugzilla dataset in `data/bugzilla_clean.csv`. To reproduce:

```bash
python run_all.py            # run all experiments and evaluate (150 samples each)
python run_all.py --n 50     # fewer samples
python run_all.py --eval-only  # re-evaluate saved results only
```

The DistilBERT model was fine-tuned with `training/bug_structure.py` (written for Google Colab).

## Project structure

```
api/          FastAPI app, model inference, SQLite helpers
frontend/     Dashboard (single index.html)
models/       Fine-tuned DistilBERT config + tokenizer (weights downloaded separately)
pipeline/     LLM extraction experiments
evaluation/   Evaluation scripts and saved results
training/     DistilBERT fine-tuning script
data/         Bugzilla datasets
schemas/      Bug schema and severity mapping
```
