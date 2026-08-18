import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import os

print("="*80)
print("OBJECTIVE 5: PARAPHRASE EVASION ANALYSIS")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
with open('models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
df = pd.read_csv('data/processed/AI_vs_Human_Data_Parapar.csv')
print(f"✓ Loaded {len(df)} rows")

# Show columns
print(f"\nColumns: {df.columns.tolist()}")

# Identify columns (using your actual column names)
text_col = 'Source'  # The text content
ground_truth_col = 'Model'  # AI or Human
paraphrased_col = 'Paraphrased'  # Yes or No

# Detector columns
detector_cols = ['ChatGPT', 'Gemini', 'Claude', 'zeroGPT']

print(f"\nText column: '{text_col}'")
print(f"Ground truth column: '{ground_truth_col}'")
print(f"Paraphrased column: '{paraphrased_col}'")
print(f"Detector columns: {detector_cols}")

# Prepare data
print("\n[2/5] Preparing data...")

# Create ground truth (1 for AI, 0 for Human)
df['ground_truth'] = (df[ground_truth_col] == 'AI').astype(int)
print(f"✓ Ground truth: {df['ground_truth'].sum()} AI, {(df['ground_truth']==0).sum()} Human")

# Create paraphrased indicator (1 for Yes, 0 for No)
df['is_paraphrased'] = (df[paraphrased_col].astype(str).str.lower() == 'yes').astype(int)
print(f"✓ Paraphrased: {df['is_paraphrased'].sum()} Yes, {(df['is_paraphrased']==0).sum()} No")

# Convert detector columns to binary (1 for AI detection)
for col in detector_cols:
    if col in df.columns:
        df[col + '_binary'] = (df[col].astype(str).str.upper() == 'AI').astype(int)
        print(f"✓ Converted '{col}' to binary")

# Run predictions
print("\n[3/5] Running predictions...")
X = vectorizer.transform(df[text_col].fillna(''))
df['my_model_pred'] = model.predict(X)
print(f"✓ Predictions complete")
print(f"  Predicted AI: {df['my_model_pred'].sum()}")
print(f"  Predicted Human: {(df['my_model_pred']==0).sum()}")

# Calculate metrics
print("\n[4/5] Calculating metrics...")

# Define detectors
detectors = {'My Model': 'my_model_pred'}
for col in detector_cols:
    binary_col = col + '_binary'
    if binary_col in df.columns:
        detectors[col] = binary_col

results = []
for name, col in detectors.items():
    if col not in df.columns:
        continue
    
    # Overall accuracy
    overall = accuracy_score(df['ground_truth'], df[col])
    
    # Paraphrased accuracy
    para_mask = df['is_paraphrased'] == 1
    if para_mask.sum() > 0:
        para_acc = accuracy_score(df[para_mask]['ground_truth'], df[para_mask][col])
        para_count = para_mask.sum()
    else:
        para_acc = np.nan
        para_count = 0
    
    # Non-paraphrased accuracy
    non_para_mask = df['is_paraphrased'] == 0
    if non_para_mask.sum() > 0:
        non_para_acc = accuracy_score(df[non_para_mask]['ground_truth'], df[non_para_mask][col])
        non_para_count = non_para_mask.sum()
    else:
        non_para_acc = np.nan
        non_para_count = 0
    
    # Accuracy drop
    if not (np.isnan(para_acc) or np.isnan(non_para_acc)):
        drop = non_para_acc - para_acc
    else:
        drop = np.nan
    
    results.append({
        'Detector': name,
        'Non-Paraphrased': non_para_acc,
        'Paraphrased': para_acc,
        'Drop': drop,
        'Overall': overall,
        'Non-Para Samples': non_para_count,
        'Para Samples': para_count
    })

summary_df = pd.DataFrame(results).round(4)
summary_df = summary_df.sort_values('Drop', ascending=False)

# Display results
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))

# Save results
os.makedirs('results', exist_ok=True)
summary_df.to_csv('results/paraphrase_analysis_summary.csv', index=False)
print("\n✓ Results saved to 'results/paraphrase_analysis_summary.csv'")

# Data breakdown
print("\n" + "="*80)
print("DATA BREAKDOWN")
print("="*80)
print(f"Total samples: {len(df)}")
print(f"AI samples: {df['ground_truth'].sum()}")
print(f"Human samples: {(df['ground_truth']==0).sum()}")
print(f"Paraphrased samples: {df['is_paraphrased'].sum()}")
print(f"Non-paraphrased samples: {(df['is_paraphrased']==0).sum()}")

# Interpretation
print("\n[5/5] Interpretation")
print("="*80)
valid = summary_df.dropna(subset=['Drop'])
if not valid.empty:
    my_row = valid[valid['Detector'] == 'My Model']
    if not my_row.empty:
        my_drop = my_row.iloc[0]['Drop']
        my_non = my_row.iloc[0]['Non-Paraphrased']
        my_para = my_row.iloc[0]['Paraphrased']
        
        print(f"\nYour Model Performance:")
        print(f"  - Non-paraphrased accuracy: {my_non:.2%}")
        print(f"  - Paraphrased accuracy: {my_para:.2%}")
        print(f"  - Accuracy drop: {my_drop:.2%}")
        
        if my_drop > 0.10:
            print("\n⚠️  Your model is VULNERABLE to paraphrasing (drop > 10%)")
        elif my_drop > 0.05:
            print("\n⚠️  Your model shows MODERATE vulnerability to paraphrasing (drop 5-10%)")
        else:
            print("\n✅  Your model shows ROBUSTNESS to paraphrasing (drop < 5%)")
    
    # Find worst performer
    worst = valid.loc[valid['Drop'].idxmax()]
    print(f"\nParaphrasing is MOST effective against: {worst['Detector']}")
    print(f"  Accuracy drop: {worst['Drop']:.2%}")
    print(f"  (Non-paraphrased: {worst['Non-Paraphrased']:.2%} → Paraphrased: {worst['Paraphrased']:.2%})")
    
    # Find best performer
    best = valid.loc[valid['Drop'].idxmin()]
    print(f"\nParaphrasing is LEAST effective against: {best['Detector']}")
    print(f"  Accuracy drop: {best['Drop']:.2%}")
    print(f"  (Non-paraphrased: {best['Non-Paraphrased']:.2%} → Paraphrased: {best['Paraphrased']:.2%})")
    
    # Overall interpretation
    avg_drop = valid['Drop'].mean()
    print(f"\nAverage accuracy drop across all detectors: {avg_drop:.2%}")
    print(f"Range of drops: {valid['Drop'].min():.2%} to {valid['Drop'].max():.2%}")

print("\n✨ Analysis complete!")
EOFcat > obj5_fixed.py << 'EOF'
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import os

print("="*80)
print("OBJECTIVE 5: PARAPHRASE EVASION ANALYSIS")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
with open('models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
df = pd.read_csv('data/processed/AI_vs_Human_Data_Parapar.csv')
print(f"✓ Loaded {len(df)} rows")

# Show columns
print(f"\nColumns: {df.columns.tolist()}")

# Identify columns (using your actual column names)
text_col = 'Source'  # The text content
ground_truth_col = 'Model'  # AI or Human
paraphrased_col = 'Paraphrased'  # Yes or No

# Detector columns
detector_cols = ['ChatGPT', 'Gemini', 'Claude', 'zeroGPT']

print(f"\nText column: '{text_col}'")
print(f"Ground truth column: '{ground_truth_col}'")
print(f"Paraphrased column: '{paraphrased_col}'")
print(f"Detector columns: {detector_cols}")

# Prepare data
print("\n[2/5] Preparing data...")

# Create ground truth (1 for AI, 0 for Human)
df['ground_truth'] = (df[ground_truth_col] == 'AI').astype(int)
print(f"✓ Ground truth: {df['ground_truth'].sum()} AI, {(df['ground_truth']==0).sum()} Human")

# Create paraphrased indicator (1 for Yes, 0 for No)
df['is_paraphrased'] = (df[paraphrased_col].astype(str).str.lower() == 'yes').astype(int)
print(f"✓ Paraphrased: {df['is_paraphrased'].sum()} Yes, {(df['is_paraphrased']==0).sum()} No")

# Convert detector columns to binary (1 for AI detection)
for col in detector_cols:
    if col in df.columns:
        df[col + '_binary'] = (df[col].astype(str).str.upper() == 'AI').astype(int)
        print(f"✓ Converted '{col}' to binary")

# Run predictions
print("\n[3/5] Running predictions...")
X = vectorizer.transform(df[text_col].fillna(''))
df['my_model_pred'] = model.predict(X)
print(f"✓ Predictions complete")
print(f"  Predicted AI: {df['my_model_pred'].sum()}")
print(f"  Predicted Human: {(df['my_model_pred']==0).sum()}")

# Calculate metrics
print("\n[4/5] Calculating metrics...")

# Define detectors
detectors = {'My Model': 'my_model_pred'}
for col in detector_cols:
    binary_col = col + '_binary'
    if binary_col in df.columns:
        detectors[col] = binary_col

results = []
for name, col in detectors.items():
    if col not in df.columns:
        continue
    
    # Overall accuracy
    overall = accuracy_score(df['ground_truth'], df[col])
    
    # Paraphrased accuracy
    para_mask = df['is_paraphrased'] == 1
    if para_mask.sum() > 0:
        para_acc = accuracy_score(df[para_mask]['ground_truth'], df[para_mask][col])
        para_count = para_mask.sum()
    else:
        para_acc = np.nan
        para_count = 0
    
    # Non-paraphrased accuracy
    non_para_mask = df['is_paraphrased'] == 0
    if non_para_mask.sum() > 0:
        non_para_acc = accuracy_score(df[non_para_mask]['ground_truth'], df[non_para_mask][col])
        non_para_count = non_para_mask.sum()
    else:
        non_para_acc = np.nan
        non_para_count = 0
    
    # Accuracy drop
    if not (np.isnan(para_acc) or np.isnan(non_para_acc)):
        drop = non_para_acc - para_acc
    else:
        drop = np.nan
    
    results.append({
        'Detector': name,
        'Non-Paraphrased': non_para_acc,
        'Paraphrased': para_acc,
        'Drop': drop,
        'Overall': overall,
        'Non-Para Samples': non_para_count,
        'Para Samples': para_count
    })

summary_df = pd.DataFrame(results).round(4)
summary_df = summary_df.sort_values('Drop', ascending=False)

# Display results
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))

# Save results
os.makedirs('results', exist_ok=True)
summary_df.to_csv('results/paraphrase_analysis_summary.csv', index=False)
print("\n✓ Results saved to 'results/paraphrase_analysis_summary.csv'")

# Data breakdown
print("\n" + "="*80)
print("DATA BREAKDOWN")
print("="*80)
print(f"Total samples: {len(df)}")
print(f"AI samples: {df['ground_truth'].sum()}")
print(f"Human samples: {(df['ground_truth']==0).sum()}")
print(f"Paraphrased samples: {df['is_paraphrased'].sum()}")
print(f"Non-paraphrased samples: {(df['is_paraphrased']==0).sum()}")

# Interpretation
print("\n[5/5] Interpretation")
print("="*80)
valid = summary_df.dropna(subset=['Drop'])
if not valid.empty:
    my_row = valid[valid['Detector'] == 'My Model']
    if not my_row.empty:
        my_drop = my_row.iloc[0]['Drop']
        my_non = my_row.iloc[0]['Non-Paraphrased']
        my_para = my_row.iloc[0]['Paraphrased']
        
        print(f"\nYour Model Performance:")
        print(f"  - Non-paraphrased accuracy: {my_non:.2%}")
        print(f"  - Paraphrased accuracy: {my_para:.2%}")
        print(f"  - Accuracy drop: {my_drop:.2%}")
        
        if my_drop > 0.10:
            print("\n⚠️  Your model is VULNERABLE to paraphrasing (drop > 10%)")
        elif my_drop > 0.05:
            print("\n⚠️  Your model shows MODERATE vulnerability to paraphrasing (drop 5-10%)")
        else:
            print("\n✅  Your model shows ROBUSTNESS to paraphrasing (drop < 5%)")
    
    # Find worst performer
    worst = valid.loc[valid['Drop'].idxmax()]
    print(f"\nParaphrasing is MOST effective against: {worst['Detector']}")
    print(f"  Accuracy drop: {worst['Drop']:.2%}")
    print(f"  (Non-paraphrased: {worst['Non-Paraphrased']:.2%} → Paraphrased: {worst['Paraphrased']:.2%})")
    
    # Find best performer
    best = valid.loc[valid['Drop'].idxmin()]
    print(f"\nParaphrasing is LEAST effective against: {best['Detector']}")
    print(f"  Accuracy drop: {best['Drop']:.2%}")
    print(f"  (Non-paraphrased: {best['Non-Paraphrased']:.2%} → Paraphrased: {best['Paraphrased']:.2%})")
    
    # Overall interpretation
    avg_drop = valid['Drop'].mean()
    print(f"\nAverage accuracy drop across all detectors: {avg_drop:.2%}")
    print(f"Range of drops: {valid['Drop'].min():.2%} to {valid['Drop'].max():.2%}")

print("\n✨ Analysis complete!")
