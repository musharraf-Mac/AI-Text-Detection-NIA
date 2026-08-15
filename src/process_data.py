import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns
from sklearn.model_selection import train_test_split

# If there is no data_p directory, this code will create
os.makedirs(os.path.join("..", "data_p", "processed"), exist_ok=True)

raw_d_path = os.path.join("..", "data_p","TXD_22.csv")
output_path = os.path.join("..", "data_p","processed", "TXD_22_cleaned.csv")

df = pd.read_csv(raw_d_path)
# print(df.head())
# print(df.tail())

df.columns=[c.strip() for c in df.columns]  # Strip whitespace from column names

df = df.rename(columns={'Answer': 'text', 'Source':'gen_source'})

df["gen_source"] = df["gen_source"].str.strip().str.lower() # Normalize the 'gen_source' column

# Derive binary author column
df['author'] = df['gen_source'].apply(lambda x: 'human' if x == 'human written' else('mixed' if ('ai-generated and human-written' in x) or ('human-written and ai-refined' in x) else 'ai'))

before = len(df)
df = df[df['author']!= 'mixed']  # Remove mixed entries
print(f"Removed {before - len(df)} mixed entries. Remaining entries: {len(df)}")

print(df["gen_source"].value_counts())

# seperate a model as unseen for objective 4 testing
unseen_model ='qwen'

held_out = df[df['gen_source'] == unseen_model]
remaining = df[df['gen_source']!= unseen_model]

# splitting
train_df, test_df = train_test_split(remaining, test_size=0.2, random_state=42, stratify=remaining['author'])
held_out.to_csv(os.path.join("..", "data_p","processed", "unseen_gen_test.csv"), index=False)

train_df["split"] = "train"
test_df["split"] = "test"
final = pd.concat([train_df, test_df], ignore_index=True)
final.to_csv(output_path, index=False)
