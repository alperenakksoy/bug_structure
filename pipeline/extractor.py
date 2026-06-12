import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_PATH = "/Users/alperenaksoy/Desktop/bug-report-structurer/data/bugzilla.csv"

SEVERITY_MAPPING = {
    "trivial": "low",
    "minor": "low",
    "normal": "medium",
    "major": "high",
    "critical": "critical",
    "blocker": "critical"
}

def extract_bug_info(report: str, model: str = "llama-3.1-8b-instant") -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a bug report analyzer. Always respond with ONLY valid JSON. No explanations, no markdown, no extra text."
            },
            {
                "role": "user",
                "content": f"""Extract structured information from this bug report.

Bug report: {report}

Return ONLY a JSON object with these fields:
- schema_type (backend, frontend, database, performance, other)
- component
- trigger_action
- error_signature
- severity (low, medium, high, critical)
"""
            }
        ],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_pipeline(n_samples: int = 20):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    print("Columns:", df.columns.tolist())  # debug

    # Her severity'den manuel örnekle
    sample_list = []
    for label in df["Severity Label"].unique():
        group = df[df["Severity Label"] == label]
        n = min(len(group), max(1, n_samples // 6))
        sample_list.append(group.sample(n, random_state=42))
    samples = pd.concat(sample_list).reset_index(drop=True)

    print("Sample columns:", samples.columns.tolist())  # debug
    print("Sample shape:", samples.shape)  # debug

    results = []

    for i, row in samples.iterrows():
        desc = row["Short Description"]
        severity_label = row["Severity Label"]
        bug_id = row["Bug ID"]

        print(f"Processing {i+1}/{len(samples)}: {desc[:50]}...")

        try:
            extracted = extract_bug_info(desc)
            extracted["bug_id"] = bug_id
            extracted["true_severity"] = SEVERITY_MAPPING.get(severity_label.strip().lower(), "medium")
            extracted["original_label"] = severity_label
            extracted["description"] = desc
            extracted["correct_severity"] = extracted["severity"] == extracted["true_severity"]
            results.append(extracted)

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue

    output_path = "/Users/alperenaksoy/Desktop/bug-report-structurer/evaluation/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        correct = sum(1 for r in results if r.get("correct_severity"))
        print(f"\n=== RESULTS ===")
        print(f"Total processed: {len(results)}")
        print(f"Severity accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    else:
        print("No results!")

    return results


if __name__ == "__main__":
    run_pipeline(n_samples=20)