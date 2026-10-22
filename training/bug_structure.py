"""

# -*- coding: utf-8 -*-

# !pip install transformers datasets scikit-learn accelerate -q



# 2. VERİYİ YÜKLE VE SÜTUN İSİMLERİNİ DÜZELT
# ============================================================
from google.colab import files
uploaded = files.upload()  # bugzilla_clean.csv dosyasını seç

import pandas as pd
df = pd.read_csv("bugzilla_clean.csv")

# Gerçek sütun isimlerini kontrol et
print("Orijinal sütunlar:", df.columns.tolist())

# Kodun beklediği isimlere eşle
df = df.rename(columns={
    "Short Description": "description",
    "Severity Label": "severity",
})

print("Yeni sütunlar:", df.columns.tolist())
print(df.shape)
print(df["severity"].value_counts())

# ============================================================
# 3. SEVERITY MAPPING + LABEL ENCODING + TRAIN/VAL/TEST SPLIT
# ============================================================
from sklearn.model_selection import train_test_split

# Bugzilla'nın native 6 etiketini, LLM deneylerindeki 4 sınıfa eşle
# (Milestone 1-3'teki mapping ile birebir aynı, karşılaştırılabilir olsun diye)
severity_map = {
    "trivial": "low", "minor": "low",
    "normal": "medium",
    "major": "high",
    "critical": "critical", "blocker": "critical",
}
df["severity"] = df["severity"].map(severity_map)

# Mapping sonrası boş/eşleşmeyen satır kaldıysa kontrol et
print("Mapping sonrası dağılım:")
print(df["severity"].value_counts())
print("Eşleşmeyen (NaN) satır sayısı:", df["severity"].isna().sum())
df = df.dropna(subset=["severity"])  # varsa eşleşmeyenleri at

labels = sorted(df["severity"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df["label"] = df["severity"].map(label2id)

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# ============================================================
# 4. TOKENIZATION
# ============================================================
from datasets import Dataset
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["description"], truncation=True, padding="max_length", max_length=64)

train_ds = Dataset.from_pandas(train_df[["description", "label"]].reset_index(drop=True)).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val_df[["description", "label"]].reset_index(drop=True)).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test_df[["description", "label"]].reset_index(drop=True)).map(tokenize, batched=True)

# ============================================================
# 5. MODEL VE TRAINER
# ============================================================
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels_true, preds),
        "macro_f1": f1_score(labels_true, preds, average="macro"),
        "weighted_f1": f1_score(labels_true, preds, average="weighted"),
    }

args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ============================================================
# 6. TEST SETİNDE DEĞERLENDİRME (raporlanacak asıl sonuç)
# ============================================================
from sklearn.metrics import classification_report, confusion_matrix

preds = trainer.predict(test_ds)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = test_ds["label"]

print("=== TEST SET SONUÇLARI (4 sınıf: low/medium/high/critical) ===")
print(compute_metrics((preds.predictions, y_true)))
print()
print(classification_report(y_true, y_pred, target_names=labels))
print("Confusion matrix (satır=gerçek, sütun=tahmin):")
print(confusion_matrix(y_true, y_pred))

# ============================================================
# 7. MODELİ KAYDET VE İNDİR
# ============================================================
trainer.save_model("./severity_distilbert_4class")
tokenizer.save_pretrained("./severity_distilbert_4class")

!zip -r severity_distilbert_4class.zip severity_distilbert_4class
files.download("severity_distilbert_4class.zip")

import numpy as np
import pandas as pd
from scipy.special import softmax

# 1. Get confidence (probability) from raw logits on the test set
preds = trainer.predict(test_ds)
logits = preds.predictions
probs = softmax(logits, axis=1)          # probability per class
y_pred = np.argmax(probs, axis=1)
confidence = np.max(probs, axis=1)       # highest probability = how confident the model is
y_true = np.array(test_ds["label"])

results_df = pd.DataFrame({
    "true_label": [id2label[i] for i in y_true],
    "pred_label": [id2label[i] for i in y_pred],
    "confidence": confidence,
    "correct": (y_pred == y_true),
})

# 2. Trade-off analysis across different confidence thresholds
print(f"{'Threshold':<10}{'Auto %':<12}{'Auto Acc':<14}{'Human Review %':<18}")
for threshold in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    auto_mask = results_df["confidence"] >= threshold
    auto_pct = auto_mask.mean() * 100
    auto_acc = results_df.loc[auto_mask, "correct"].mean() * 100 if auto_mask.sum() > 0 else float("nan")
    review_pct = 100 - auto_pct
    print(f"{threshold:<10}{auto_pct:<12.1f}{auto_acc:<14.1f}{review_pct:<18.1f}")

# 3. Example: what happens if we pick threshold = 0.5?
threshold = 0.5
auto_mask = results_df["confidence"] >= threshold
print(f"\nThreshold={threshold}:")
print(f"  Auto-decided: {auto_mask.sum()} / {len(results_df)} ({auto_mask.mean()*100:.1f}%)")
print(f"  Accuracy in this group: {results_df.loc[auto_mask,'correct'].mean()*100:.1f}%")
print(f"  Sent to human review: {(~auto_mask).sum()} ({(~auto_mask).mean()*100:.1f}%)")
print(f"  Accuracy of the reviewed (low-confidence) group: {results_df.loc[~auto_mask,'correct'].mean()*100:.1f}%")

import matplotlib.pyplot as plt

thresholds = np.linspace(0, 0.95, 20)
coverage, acc = [], []
for t in thresholds:
    mask = results_df["confidence"] >= t
    coverage.append(mask.mean() * 100)
    acc.append(results_df.loc[mask, "correct"].mean() * 100 if mask.sum() > 0 else np.nan)

fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(thresholds, coverage, label="Auto-decided %", color="tab:blue")
ax1.set_xlabel("Confidence threshold")
ax1.set_ylabel("Auto-decided %", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(thresholds, acc, label="Accuracy", color="tab:red")
ax2.set_ylabel("Accuracy %", color="tab:red")
plt.title("Confidence threshold: coverage vs. accuracy trade-off")
plt.tight_layout()
plt.savefig("confidence_tradeoff.png", dpi=150)
plt.show()


"""