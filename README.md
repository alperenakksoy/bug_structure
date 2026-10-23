# Bug Report Structurer 🐞

A hybrid AI-powered backend system that automatically analyzes, categorizes, and extracts structured data from raw software bug reports.

## Architecture

- **Local DistilBERT (fine-tuned):** Runs locally to determine the bug's `severity` (low, medium, high, critical) and calculates a confidence score.
- **Groq API (LLMs):** Handles complex text extraction to identify the Schema Type (Backend, Frontend, Database, etc.), Component, and Trigger Actions.
- **FastAPI:** Provides a fast, modern API backend that serves the models and processes incoming reports.

---

## 🚀 Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/alperenakksoy/bug_structure.git
cd bug_structure
```

### 2. Download the local AI model (crucial step)

To keep the GitHub repository lightweight, the heavy model weights (`model.safetensors`) are hosted externally. The system will crash if this file is missing.

1. Download `model.safetensors` from this link: [Download model.safetensors](https://drive.google.com/file/d/1Q6jheVj332YWVLrwisqX4o47xAGyX4H5/view?usp=drive_link)
2. Move the file into this directory of the project: `models/severity_distilbert_4class/`

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

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

### 5. Configure API keys

The application needs a Groq API key to communicate with the Llama models.

1. Create a file named `.env` in the project root.
2. Add your key:

```env
GROQ_API_KEY=your_actual_api_key_here
```

### 6. Run the server

Start the FastAPI backend with Uvicorn. The `.venv` folder is excluded from the reload watcher to avoid unnecessary restarts.

```bash
uvicorn api.main:app --reload --reload-exclude ".venv"
```

### 7. Test the application

Open the frontend dashboard at <http://localhost:8000/dashboard> in your browser, paste any raw software bug complaint into the interface, and click **Analyze** to see the severity and extracted technical data.
