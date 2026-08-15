import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns
from sklearn.model_selection import train_test_split

# If there is no data_p directory, this code will create
os.makedirs(os.path.join("..", "data_p", "processed"), exist_ok=True)

raw_d_path = os.path.join("..", "data_p","ai_vs_human_text_2026_gpt_gemini.csv")
output_path = os.path.join("..", "data_p","processed", "ai_h_cln.csv")

df = pd.read_csv(raw_d_path)
# print(df.head())
# print(df.tail())

df.columns=[c.strip() for c in df.columns]  # Strip whitespace from column names

df = df.rename(columns={'text_content': 'text', 'label':'author'})

df["author"] = df["author"].str.strip().str.lower() # Normalize the 'gen_source' column

before = len(df)
print(f"Before removing duplicates: {before}")
# Remove duplicates
df = df.drop_duplicates(subset=['text'])
print(f"Dropped {before - len(df)} duplicate rows.")

print(df["author"].value_counts())

# splitting
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['author'])
train_df["split"] = "train"
test_df["split"] = "test"

final = pd.concat([train_df, test_df], ignore_index=True)
final.to_csv(output_path, index=False)

