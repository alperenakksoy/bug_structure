import pandas as pd

df = pd.read_csv('schema_review.csv')

cropped_df = df[['bug_id', 'description','my_review']]

cropped_df.to_csv('cropped_data.csv', index=False)

print("CSV successfully cropped and saved!")