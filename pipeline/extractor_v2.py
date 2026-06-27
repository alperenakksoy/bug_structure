"""
Bug report structured information extractor.

Supports three models via Groq API:
  llama_8b  → llama-3.1-8b-instant
  llama_70b → llama-3.3-70b-versatile
  qwen      → qwen-qwq-32b

Two prompt strategies:
  zero_shot — direct extraction with schema description only
  few_shot  — three labeled examples prepended before the target report

Output per record includes all extracted fields plus:
  json_valid       (bool) — whether the model returned parseable JSON
  hallucinated     (bool) — whether any field value is implausibly generic
  correct_severity (bool) — predicted severity == true severity
"""

import os
import json
import time
import csv
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_PATH = "data/bugzilla_clean.csv"

SEVERITY_MAPPING = {
    "trivial":  "low",
    "minor":    "low",
    "normal":   "medium",
    "major":    "high",
    "critical": "critical",
    "blocker":  "critical",
}

VALID_SCHEMA_TYPES = {"backend", "frontend", "database", "performance", "other"}
VALID_SEVERITIES   = {"low", "medium", "high", "critical"}

MODELS = {
    "llama_8b":  "llama-3.1-8b-instant",
    "llama_70b": "llama-3.3-70b-versatile",
    "qwen":      "qwen-qwq-32b",
}

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a bug report analyzer. "
    "Always respond with ONLY valid JSON — no markdown fences, no explanations."
)

FIELD_DESCRIPTION = """\
Return ONLY a JSON object with exactly these fields:
  schema_type     — one of: backend, frontend, database, performance, other
  component       — the software component or file affected (string)
  trigger_action  — what action or event caused the bug (string)
  error_signature — the exact error message or observable symptom (string)
  severity        — one of: low, medium, high, critical"""

FEW_SHOT_EXAMPLES = """\
Examples:

Bug report: "LDAP user login failure: Can't locate object method 'realname'. Happens on token refresh."
Output: {"schema_type": "backend", "component": "auth_service", "trigger_action": "token refresh", "error_signature": "Can't locate object method 'realname'", "severity": "critical"}

Bug report: "Wrong colspan for summarize time link on show_bug.cgi when viewing on Firefox 112."
Output: {"schema_type": "frontend", "component": "show_bug.cgi", "trigger_action": "page rendering", "error_signature": "wrong colspan", "severity": "low"}

Bug report: "Database connection timeout when running complex queries on large datasets."
Output: {"schema_type": "database", "component": "query_engine", "trigger_action": "complex query execution", "error_signature": "connection timeout", "severity": "high"}

"""


def build_user_prompt(report: str, prompt_type: str) -> str:
    if prompt_type == "zero_shot":
        return f"Extract structured information from this bug report.\n\nBug report: {report}\n\n{FIELD_DESCRIPTION}"
    else:
        return (
            f"Extract structured information from bug reports.\n\n"
            f"{FEW_SHOT_EXAMPLES}"
            f"Now extract from this bug report:\n"
            f"Bug report: {report}\n\n"
            f"{FIELD_DESCRIPTION}"
        )


# ── Field-level validation ───────────────────────────────────────────────────

HALLUCINATION_TOKENS = {"unknown", "n/a", "none", "not specified", "not provided", "na", ""}


def _is_hallucinated(value: str) -> bool:
    return str(value).strip().lower() in HALLUCINATION_TOKENS


def field_extraction_metrics(record: dict) -> dict:
    """
    Computes per-record structural quality metrics for the extracted fields.
    Does not require ground-truth field annotations — measures validity and
    completeness only.
    """
    fields = ["schema_type", "component", "trigger_action", "error_signature", "severity"]
    present = {f: f in record and record[f] not in (None, "") for f in fields}
    hallucinated_fields = {f: _is_hallucinated(record.get(f, "")) for f in fields}
    schema_type_valid = record.get("schema_type", "") in VALID_SCHEMA_TYPES
    severity_valid    = record.get("severity", "")    in VALID_SEVERITIES

    completeness = sum(present.values()) / len(fields)
    hallucination_any = any(hallucinated_fields.values())

    return {
        "field_completeness":   round(completeness, 4),
        "schema_type_valid":    schema_type_valid,
        "severity_field_valid": severity_valid,
        "hallucinated":         hallucination_any,
        "hallucinated_fields":  [f for f, h in hallucinated_fields.items() if h],
    }


# ── Single extraction call ───────────────────────────────────────────────────

def extract_bug_info(report: str, model_key: str = "llama_8b", prompt_type: str = "zero_shot") -> tuple[dict, bool]:
    """
    Returns (extracted_dict, json_valid).
    On parse failure returns a skeleton dict with json_valid=False.
    """
    time.sleep(1)

    response = client.chat.completions.create(
        model=MODELS[model_key],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(report, prompt_type)},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    # strip optional markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()

    try:
        parsed = json.loads(raw)
        return parsed, True
    except json.JSONDecodeError:
        # Try extracting a JSON object substring
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                return parsed, True
            except json.JSONDecodeError:
                pass
        return {
            "schema_type": "", "component": "", "trigger_action": "",
            "error_signature": "", "severity": "",
            "_raw_response": raw[:300],
        }, False


# ── Experiment runner ────────────────────────────────────────────────────────

def load_dataset(n_samples: int) -> list[dict]:
    """Stratified sample — equal representation of each severity label."""
    rows: list[dict] = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        by_severity: dict[str, list] = {}
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            label = row["Severity Label"].lower()
            by_severity.setdefault(label, []).append(row)

    per_class = max(1, n_samples // len(by_severity))
    import random
    rng = random.Random(42)
    for label, group in by_severity.items():
        sampled = rng.sample(group, min(per_class, len(group)))
        rows.extend(sampled)
    return rows[:n_samples]


def run_experiment(n_samples: int = 150, model_key: str = "llama_8b", prompt_type: str = "zero_shot"):
    suffix = f"{model_key}_{prompt_type}"
    out_path = f"evaluation/results_{suffix}.json"

    # Remap qwen key to include size in filename for clarity
    file_key = "qwen_32b" if model_key == "qwen" else model_key
    out_path = f"evaluation/results_{file_key}_{prompt_type}.json"

    samples = load_dataset(n_samples)
    print(f"\nModel: {MODELS[model_key]} | Prompt: {prompt_type} | Samples: {len(samples)}")

    results = []
    for i, row in enumerate(samples):
        desc  = row["Short Description"]
        label = row["Severity Label"].strip().lower()
        bug_id = row["Bug ID"]

        true_sev = SEVERITY_MAPPING.get(label, "medium")

        print(f"  [{i+1:>3}/{len(samples)}] {desc[:60]}...")

        try:
            extracted, json_valid = extract_bug_info(desc, model_key=model_key, prompt_type=prompt_type)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            continue

        record = {
            **extracted,
            "bug_id":          bug_id,
            "description":     desc,
            "original_label":  label,
            "true_severity":   true_sev,
            "model":           model_key,
            "model_id":        MODELS[model_key],
            "prompt_type":     prompt_type,
            "json_valid":      json_valid,
            "correct_severity": json_valid and extracted.get("severity") == true_sev,
        }
        record.update(field_extraction_metrics(extracted))
        results.append(record)

    os.makedirs("evaluation", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        correct = sum(1 for r in results if r.get("correct_severity"))
        valid   = sum(1 for r in results if r.get("json_valid"))
        print(f"\n  === {model_key} | {prompt_type} ===")
        print(f"  Total:            {len(results)}")
        print(f"  JSON valid:       {valid}/{len(results)} ({valid/len(results)*100:.1f}%)")
        print(f"  Severity accuracy:{correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
        print(f"  Saved → {out_path}")
    else:
        print("  No results produced.")

    return results
