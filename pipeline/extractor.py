import os
import json
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_PATH = "/Users/alperenaksoy/Desktop/bug-report-structurer/data/bugzilla_clean.csv"

SEVERITY_MAPPING = {
    "trivial": "low",
    "minor": "low",
    "normal": "medium",
    "major": "high",
    "critical": "critical",
    "blocker": "critical"
}

MODELS = {
    "llama": "llama-3.1-8b-instant",
    "gemma": "llama-3.3-70b-versatile"
}

def extract_bug_info(report: str, model: str = "llama", prompt_type: str = "zero_shot") -> dict:
    time.sleep(1)

    if prompt_type == "zero_shot":
        user_content = f"""Extract structured information from this bug report.

Bug report: {report}

Return ONLY a JSON object with these fields:
- schema_type (backend, frontend, database, performance, other)
- component
- trigger_action
- error_signature
- severity (low, medium, high, critical)
"""
    else:
        user_content = f"""Extract structured information from bug reports. Here are examples:

Example 1:
Bug report: "LDAP user login failure: Can't locate object method 'realname'. Happens on token refresh."
Output: {{"schema_type": "backend", "component": "auth_service", "trigger_action": "token refresh", "error_signature": "Can't locate object method 'realname'", "severity": "critical"}}

Example 2:
Bug report: "Wrong colspan for summarize time link on show_bug.cgi when viewing on Firefox 112."
Output: {{"schema_type": "frontend", "component": "show_bug.cgi", "trigger_action": "page rendering", "error_signature": "wrong colspan", "severity": "low"}}

Example 3:
Bug report: "Database connection timeout when running complex queries on large datasets."
Output: {{"schema_type": "database", "component": "query_engine", "trigger_action": "complex query execution", "error_signature": "connection timeout", "severity": "high"}}

Now extract from this bug report:
Bug report: {report}

Return ONLY a JSON object with the same fields.
"""

    response = client.chat.completions.create(
        model=MODELS[model],
        messages=[
            {"role": "system", "content": "You are a bug report analyzer. Always respond with ONLY valid JSON. No explanations, no markdown, no extra text."},
            {"role": "user", "content": user_content}
        ],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_pipeline(n_samples: int = 30, model: str = "llama", prompt_type: str = "zero_shot"):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    sample_list = []
    for label in df["Severity Label"].unique():
        group = df[df["Severity Label"] == label]
        n = min(len(group), max(1, n_samples // 6))
        sample_list.append(group.sample(n, random_state=42))
    samples = pd.concat(sample_list).reset_index(drop=True)

    print(f"Model: {model} | Prompt: {prompt_type} | Samples: {len(samples)}\n")

    results = []

    for i, row in samples.iterrows():
        desc = row["Short Description"]
        severity_label = row["Severity Label"]
        bug_id = row["Bug ID"]

        print(f"Processing {i+1}/{len(samples)}: {desc[:50]}...")

        try:
            extracted = extract_bug_info(desc, model=model, prompt_type=prompt_type)
            extracted["bug_id"] = bug_id
            extracted["true_severity"] = SEVERITY_MAPPING.get(severity_label.strip().lower(), "medium")
            extracted["original_label"] = severity_label
            extracted["description"] = desc
            extracted["model"] = model
            extracted["prompt_type"] = prompt_type
            extracted["correct_severity"] = extracted["severity"] == extracted["true_severity"]
            results.append(extracted)

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue

    output_path = f"/Users/alperenaksoy/Desktop/bug-report-structurer/evaluation/results_{model}_{prompt_type}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        correct = sum(1 for r in results if r.get("correct_severity"))
        print(f"\n=== RESULTS ===")
        print(f"Model: {model} | Prompt: {prompt_type}")
        print(f"Total processed: {len(results)}")
        print(f"Severity accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    else:
        print("No results!")

    return results


if __name__ == "__main__":
    # Experiment A: Llama zero-shot
    run_pipeline(n_samples=30, model="llama", prompt_type="zero_shot")
    # Experiment B: Gemma zero-shot
    run_pipeline(n_samples=30, model="gemma", prompt_type="zero_shot")
    # Experiment C: Llama few-shot
    run_pipeline(n_samples=30, model="llama", prompt_type="few_shot")
    # Experiment D: Gemma few-shot
    run_pipeline(n_samples=30, model="gemma", prompt_type="few_shot")