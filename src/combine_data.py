import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df1 = pd.read_csv("../data_p/processed/gh_data_processed.csv")
df2 = pd.read_csv("../data_p/processed/TXD_22_cleaned.csv")
df3 = pd.read_csv("../data_p/processed/ai_h_cln.csv")

if 'gen_source' not in df1.columns:
    df1['gen_source'] = 'unknown'
if 'domain' not in df2.columns:
    df2['domain'] = 'unknown'

cols = ['text', 'author', 'gen_source', 'domain']
df1 = df1[[c for c in cols if c in df1.columns]]
df2 = df2[[c for c in cols if c in df2.columns]]
df3 = df3[[c for c in cols if c in df3.columns]]

df1["source_dataset"] = "ai_h"
df2["source_dataset"] = "txd_22"
df3["source_dataset"] = "gpt_t2"

combined = pd.concat([df1, df2, df3], ignore_index=True)
print(f"Combined shape: {combined.shape}")

# Dropping duplicates based on the 'text' column
before = len(combined)
combined = combined.drop_duplicates(subset=['text'])
print(f"Dropped {before - len(combined)} duplicates")

# class balance
print(combined['author'].value_counts(normalize=True))

# source vs label
print(pd.crosstab(combined['source_dataset'], combined['author']))

# domain vs label
print(pd.crosstab(combined['domain'], combined['author']))

train_df, test_df = train_test_split(
    combined, test_size=0.2, random_state=42, stratify=combined["author"]
)
train_df["split"] = "train"
test_df["split"] = "test"

final = pd.concat([train_df, test_df], ignore_index=True)
final.to_csv("../data_p/processed/dataset_comb_train.csv", index=False)
print("Saved. Final shape:", final.shape)



