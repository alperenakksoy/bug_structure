import json
import csv
import os

EVAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'subset_for_manual_validation.csv')

RESULT_FILES = {
    'llama_70b_zero_shot': 'results_llama_70b_zero_shot.json',
    'llama_70b_few_shot': 'results_llama_70b_few_shot.json',
    'llama_8b_zero_shot': 'results_llama_8b_zero_shot.json',
    'llama_8b_few_shot': 'results_llama_8b_few_shot.json',
}

# Load all results, keyed by bug_id
all_results = {}
for label, fname in RESULT_FILES.items():
    with open(os.path.join(EVAL_DIR, fname)) as f:
        for item in json.load(f):
            bug_id = item['bug_id']
            if bug_id not in all_results:
                all_results[bug_id] = {
                    'bug_id': bug_id,
                    'description': item['description'],
                    'original_severity_label': item['original_label'],
                    'true_severity': item['true_severity'],
                }
            all_results[bug_id][f'{label}__schema_type'] = item.get('schema_type', '')
            all_results[bug_id][f'{label}__component'] = item.get('component', '')
            all_results[bug_id][f'{label}__trigger_action'] = item.get('trigger_action', '')
            all_results[bug_id][f'{label}__error_signature'] = item.get('error_signature', '')
            all_results[bug_id][f'{label}__severity'] = item.get('severity', '')

rows = list(all_results.values())
rows.sort(key=lambda r: r['bug_id'])

MANUAL_FIELDS = [
    'manual__schema_type',
    'manual__component',
    'manual__trigger_action',
    'manual__error_signature',
    'manual__severity',
]

LLM_COLS = []
for label in RESULT_FILES:
    for field in ['schema_type', 'component', 'trigger_action', 'error_signature', 'severity']:
        LLM_COLS.append(f'{label}__{field}')

fieldnames = (
    ['bug_id', 'description', 'original_severity_label', 'true_severity']
    + MANUAL_FIELDS
    + LLM_COLS
)

with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        for mf in MANUAL_FIELDS:
            row.setdefault(mf, '')
        writer.writerow(row)

print(f'Written {len(rows)} rows to {OUT_PATH}')
print(f'Columns: {fieldnames}')
