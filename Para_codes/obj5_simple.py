import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import os

print("="*80)
print("OBJECTIVE 5: PARAPHRASE EVASION ANALYSIS (SIMPLIFIED)")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
df = pd.read_csv('data/processed/AI_vs_Human_Data_Parapar.csv')
print(f"✓ Loaded {len(df)} rows")

# Load model
print("\n[2/4] Loading model and vectorizer...")
with open('models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print("✓ Model and vectorizer loaded")

# Find columns
text_col = df.columns[0]
print(f"\n[3/4] Processing data...")
print(f"Text column: '{text_col}'")
print(f"Columns: {df.columns.tolist()}")

# Create ground truth
df['ground_truth'] = (df['author'] == 'AI').astype(int)

# Create paraphrased indicator
if 'paraphrased' in df.columns:
    df['is_paraphrased'] = (df['paraphrased'].astype(str).str.lower() == 'yes').astype(int)
    print(f"Paraphrased samples: {df['is_paraphrased'].sum()}")
    print(f"Non-paraphrased samples: {(df['is_paraphrased']==0).sum()}")
else:
    print("⚠️ No 'paraphrased' column found!")
    exit()

# Run predictions
print("\n[4/4] Running predictions...")
X = vectorizer.transform(df[text_col].fillna(''))
df['my_pred'] = model.predict(X)
print(f"✓ Predictions complete")
print(f"  Predicted AI: {df['my_pred'].sum()}")
print(f"  Predicted Human: {(df['my_pred']==0).sum()}")

# Calculate accuracies
print("\n" + "="*80)
print("RESULTS")
print("="*80)

# Overall
overall_acc = accuracy_score(df['ground_truth'], df['my_pred'])
print(f"Overall Accuracy: {overall_acc:.2%}")

# Paraphrased
para_mask = df['is_paraphrased'] == 1
if para_mask.sum() > 0:
    para_acc = accuracy_score(df[para_mask]['ground_truth'], df[para_mask]['my_pred'])
    print(f"Paraphrased Accuracy: {para_acc:.2%} ({para_mask.sum()} samples)")
else:
    para_acc = None

# Non-paraphrased
non_para_mask = df['is_paraphrased'] == 0
if non_para_mask.sum() > 0:
    non_para_acc = accuracy_score(df[non_para_mask]['ground_truth'], df[non_para_mask]['my_pred'])
    print(f"Non-Paraphrased Accuracy: {non_para_acc:.2%} ({non_para_mask.sum()} samples)")
else:
    non_para_acc = None

# Accuracy drop
if para_acc is not None and non_para_acc is not None:
    drop = non_para_acc - para_acc
    print(f"\nAccuracy Drop: {drop:.2%}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    if drop > 0.10:
        print("⚠️ Your model is VULNERABLE to paraphrasing (drop > 10%)")
    elif drop > 0.05:
        print("⚠️ Your model shows MODERATE vulnerability to paraphrasing (drop 5-10%)")
    else:
        print("✅ Your model shows ROBUSTNESS to paraphrasing (drop < 5%)")

print("\n✨ Analysis complete!")
