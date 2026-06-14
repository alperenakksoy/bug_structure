import pandas as pd

df = pd.read_csv("/Users/alperenaksoy/Desktop/bug-report-structurer/data/bugzilla.csv")
df.columns = df.columns.str.strip()

# 1. Çok kısa açıklamaları sil
df = df[df["Short Description"].str.len() >= 10]

# 2. Her severity'den max 2000 örnek al (dengeleme)
sample_list = []
for label in df["Severity Label"].unique():
    group = df[df["Severity Label"] == label]
    n = min(len(group), 2000)
    sample_list.append(group.sample(n, random_state=42))
balanced = pd.concat(sample_list).reset_index(drop=True)

# 3. Kaydet
balanced.to_csv("/Users/alperenaksoy/Desktop/bug-report-structurer/data/bugzilla_clean.csv", index=False)

print("Orijinal:", df.shape)
print("Temizlenmiş:", balanced.shape)
print("\nSeverity dağılımı:")
print(balanced["Severity Label"].value_counts())