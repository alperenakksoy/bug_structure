import pandas as pd

df = pd.read_csv("data/bugzilla.csv")

# Severity dağılımı
print("=== SEVERITY LABELS ===")
print(df["Severity Label"].value_counts())

# Project dağılımı
print("\n=== PROJECTS ===")
print(df["Project"].value_counts())

# Resolution dağılımı
print("\n=== RESOLUTION STATUS ===")
print(df["Resolution Status"].value_counts())

# Açıklama uzunluğu
df["desc_length"] = df["Short Description"].str.len()
print("\n=== DESCRIPTION LENGTH ===")
print(df["desc_length"].describe())

# Örnek bug reportlar
print("\n=== SAMPLE REPORTS ===")
for _, row in df.sample(5, random_state=42).iterrows():
    print(f"[{row['Severity Label']}] {row['Short Description']}")
    print()


import pandas as pd

df = pd.read_csv("data/bugzilla.csv")

print(repr(df.columns.tolist()))