import os
import json
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
from database import insert_bug

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "severity_distilbert_4class"))

severity_classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

CONFIDENCE_THRESHOLD = 0.7

EXTRACTION_PROMPT = """You are a bug report analysis assistant. Given a raw bug description, extract the following fields as a single JSON object:

- schema_type: one of "backend", "frontend", "database", "performance", "other"
- component: the part of the software affected by the bug
- trigger_action: the action or event that caused the bug
- error_signature: the exact error message or observed symptom

Respond with ONLY the JSON object, no other text.

Bug description: {description}
"""

def predict_severity(description: str) -> dict:
    result = severity_classifier(description)[0]
    return {
        "severity": result["label"],
        "confidence": round(result["score"], 4),
        "needs_human_review": result["score"] < CONFIDENCE_THRESHOLD,
    }

def extract_fields(description: str) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT.format(description=description)}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        print("GROQ RAW OUTPUT:", raw)  # debug için
        return json.loads(raw)
    except Exception as e:
        print("EXTRACTION ERROR:", repr(e))  # debug için — hatayı terminalde gör
        return {"schema_type": None, "component": None, "trigger_action": None, "error_signature": None}

def analyze_bug(description: str) -> dict:
    severity_result = predict_severity(description)
    extracted = extract_fields(description)

    result = {
        "schema_type": extracted.get("schema_type"),
        "component": extracted.get("component"),
        "trigger_action": extracted.get("trigger_action"),
        "error_signature": extracted.get("error_signature"),
        "severity": severity_result["severity"],
        "severity_confidence": severity_result["confidence"],
        "needs_human_review": severity_result["needs_human_review"],
    }

    insert_bug(description, result)  # veritabanına kaydet
    return result