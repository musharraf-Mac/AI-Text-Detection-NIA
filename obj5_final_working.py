# obj5_final_working.py - Fixed version with proper data cleaning

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("AI Detection - Paraphrase Analysis")
print("="*60)

# STEP 1: Load the data
print("\n[1] Loading data...")
data_path = 'data/processed/AI_vs_Human_Data_Parapar.csv'

try:
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit()

# STEP 2: Load model and vectorizer
print("\n[2] Loading model and vectorizer...")
try:
    with open('models/baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ Model and vectorizer loaded successfully")
    print(f"  Model type: {type(model)}")
    if hasattr(model, 'classes_'):
        print(f"  Model classes: {model.classes_}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit()

# STEP 3: Define column mappings based on actual data
print("\n[3] Setting up column mappings...")
COLUMN_MAPPINGS = {
    'text': 'Answer',           # Column containing the text
    'author': 'Source',         # Column indicating AI or Human
    'paraphrased': 'Paraphrased', # Column indicating if paraphrased
    'detectors': ['ChatGPT', 'Gemini', 'Claude', 'zeroGPT']  # Detector columns
}

print(f"  Using '{COLUMN_MAPPINGS['text']}' as text column")
print(f"  Using '{COLUMN_MAPPINGS['author']}' as author column")
print(f"  Using '{COLUMN_MAPPINGS['paraphrased']}' as paraphrased column")
print(f"  Detector columns: {COLUMN_MAPPINGS['detectors']}")

# STEP 4: Clean the data
print("\n[4] Cleaning data...")

# Clean the paraphrased column - handle NaN values
df[COLUMN_MAPPINGS['paraphrased']] = df[COLUMN_MAPPINGS['paraphrased']].fillna('No')
print(f"  Cleaned paraphrased column: {df[COLUMN_MAPPINGS['paraphrased']].unique()}")

# Clean detector columns - handle NaN values and whitespace
for detector in COLUMN_MAPPINGS['detectors']:
    if detector in df.columns:
        # Fill NaN with 'Human' (default)
        df[detector] = df[detector].fillna('Human')
        # Strip whitespace
        df[detector] = df[detector].astype(str).str.strip()
        print(f"  Cleaned {detector}: {df[detector].unique()}")

# STEP 5: Run my model predictions
print("\n[5] Running my model predictions...")
try:
    # Fill NaN values in text column and ensure string type
    text_data = df[COLUMN_MAPPINGS['text']].fillna('').astype(str)
    
    # Transform text using vectorizer
    X = vectorizer.transform(text_data)
    print(f"  Transformed data shape: {X.shape}")
    
    # Get predictions (returns strings like 'ai' or 'human')
    predictions = model.predict(X)
    print(f"  Raw predictions sample: {predictions[:5]}")
    
    # Map string predictions to binary (0=human, 1=ai)
    if hasattr(model, 'classes_'):
        print(f"  Model classes: {model.classes_}")
        # Map based on model classes
        class_mapping = {model.classes_[0]: 0, model.classes_[1]: 1}
        predictions_binary = np.array([class_mapping.get(pred, 0) for pred in predictions])
    else:
        # Fallback mapping
        predictions_binary = np.array([1 if str(pred).lower() == 'ai' else 0 for pred in predictions])
    
    df['my_model_pred'] = predictions_binary
    
    print(f"✓ Predictions completed")
    pred_counts = np.bincount(predictions_binary)
    print(f"  Distribution: Human={pred_counts[0] if len(pred_counts) > 0 else 0}, AI={pred_counts[1] if len(pred_counts) > 1 else 0}")
    
except Exception as e:
    print(f"✗ Error in predictions: {e}")
    import traceback
    traceback.print_exc()
    exit()

# STEP 6: Prepare data for analysis
print("\n[6] Preparing data for analysis...")

# Convert author to binary (0=Human, 1=AI)
author_values = df[COLUMN_MAPPINGS['author']].unique()
print(f"  Author values: {author_values}")

# Map to binary - assuming 'AI' means 1 and 'Human' means 0
df['true_label'] = df[COLUMN_MAPPINGS['author']].apply(
    lambda x: 1 if str(x).strip().upper() == 'AI' else 0
)

# Convert paraphrased to boolean
df['is_paraphrased'] = df[COLUMN_MAPPINGS['paraphrased']].apply(
    lambda x: True if str(x).strip().upper() == 'YES' else False
)

print(f"✓ Data prepared")
print(f"  True labels: Human={sum(df['true_label']==0)}, AI={sum(df['true_label']==1)}")
print(f"  Paraphrased: Yes={sum(df['is_paraphrased']==True)}, No={sum(df['is_paraphrased']==False)}")

# STEP 7: Calculate my model accuracy by group
print("\n[7] Calculating my model accuracy...")
my_results = {}

for group in [True, False]:
    group_data = df[df['is_paraphrased'] == group]
    if len(group_data) > 0:
        try:
            acc = accuracy_score(group_data['true_label'], group_data['my_model_pred'])
            my_results[group] = {'accuracy': acc, 'count': len(group_data)}
            label = "Paraphrased" if group else "Non-paraphrased"
            print(f"  My Model - {label}: {acc:.2%} ({len(group_data)} samples)")
        except Exception as e:
            label = "Paraphrased" if group else "Non-paraphrased"
            print(f"  Error calculating accuracy for {label}: {e}")

# STEP 8: Analyze commercial detectors
print("\n[8] Analyzing commercial detectors...")
detector_results = {}

for detector in COLUMN_MAPPINGS['detectors']:
    if detector in df.columns:
        print(f"\n  {detector}:")
        
        # Convert detector to binary (AI=1, Human=0)
        # First clean the values
        df[f'{detector}_clean'] = df[detector].astype(str).str.strip().str.upper()
        
        # Map to binary
        df[f'{detector}_binary'] = df[f'{detector}_clean'].apply(
            lambda x: 1 if x == 'AI' else 0
        )
        
        # Check distribution
        binary_counts = df[f'{detector}_binary'].value_counts()
        print(f"    Binary distribution: {dict(binary_counts)}")
        
        detector_results[detector] = {}
        for group in [True, False]:
            group_data = df[df['is_paraphrased'] == group]
            if len(group_data) > 0:
                try:
                    acc = accuracy_score(group_data['true_label'], group_data[f'{detector}_binary'])
                    detector_results[detector][group] = {'accuracy': acc, 'count': len(group_data)}
                    label = "Paraphrased" if group else "Non-paraphrased"
                    print(f"    {label}: {acc:.2%} ({len(group_data)} samples)")
                except Exception as e:
                    label = "Paraphrased" if group else "Non-paraphrased"
                    print(f"    Error calculating accuracy for {label}: {e}")
    else:
        print(f"  ⚠ {detector} not found in data")

# STEP 9: Create summary table
print("\n[9] Creating summary table...")

summary_data = {
    'Detector': [],
    'Paraphrased Accuracy': [],
    'Non-Paraphrased Accuracy': [],
    'Accuracy Drop': [],
    'Paraphrased Count': [],
    'Non-Paraphrased Count': []
}

# Add my model
if my_results:
    summary_data['Detector'].append('My Model')
    summary_data['Paraphrased Accuracy'].append(my_results.get(True, {}).get('accuracy', np.nan))
    summary_data['Non-Paraphrased Accuracy'].append(my_results.get(False, {}).get('accuracy', np.nan))
    summary_data['Paraphrased Count'].append(my_results.get(True, {}).get('count', 0))
    summary_data['Non-Paraphrased Count'].append(my_results.get(False, {}).get('count', 0))

# Add commercial detectors
for detector, results in detector_results.items():
    summary_data['Detector'].append(detector)
    summary_data['Paraphrased Accuracy'].append(results.get(True, {}).get('accuracy', np.nan))
    summary_data['Non-Paraphrased Accuracy'].append(results.get(False, {}).get('accuracy', np.nan))
    summary_data['Paraphrased Count'].append(results.get(True, {}).get('count', 0))
    summary_data['Non-Paraphrased Count'].append(results.get(False, {}).get('count', 0))

# Calculate accuracy drops
summary_data['Accuracy Drop'] = [
    (non_para - para) if not (np.isnan(para) or np.isnan(non_para)) else np.nan
    for para, non_para in zip(
        summary_data['Paraphrased Accuracy'],
        summary_data['Non-Paraphrased Accuracy']
    )
]

summary_df = pd.DataFrame(summary_data)

# Format percentages
for col in ['Paraphrased Accuracy', 'Non-Paraphrased Accuracy', 'Accuracy Drop']:
    summary_df[col] = summary_df[col].apply(
        lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "N/A"
    )

# STEP 10: Display results
print("\n" + "="*60)
print("SUMMARY TABLE: Accuracy on Paraphrased vs Non-Paraphrased Text")
print("="*60)
print(summary_df.to_string(index=False))

# STEP 11: Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

# Calculate average drop
drop_values = []
for _, row in summary_df.iterrows():
    if row['Accuracy Drop'] != "N/A":
        drop = float(row['Accuracy Drop'].replace('%', ''))
        drop_values.append(drop)

if drop_values:
    avg_drop = np.mean(drop_values)
    print(f"• On average, all detectors show an accuracy drop of {avg_drop:.1f}% when text has been paraphrased.")
    
    if avg_drop > 10:
        print(f"• Paraphrasing appears to be an EFFECTIVE evasion technique, reducing detection accuracy by {avg_drop:.1f}% across all tools.")
    else:
        print(f"• Paraphrasing has a MODEST effect ({avg_drop:.1f}% drop), suggesting current detectors are relatively robust to paraphrasing.")
    
    # Find best and worst performers on paraphrased text
    if len(summary_df) > 0:
        para_accs = []
        for idx, row in summary_df.iterrows():
            if row['Paraphrased Accuracy'] != "N/A":
                para_accs.append((idx, row['Detector'], float(row['Paraphrased Accuracy'].replace('%', ''))))
        
        if para_accs:
            best = max(para_accs, key=lambda x: x[2])
            worst = min(para_accs, key=lambda x: x[2])
            print(f"• Best performer on paraphrased text: {best[1]} ({best[2]:.1f}%)")
            print(f"• Worst performer on paraphrased text: {worst[1]} ({worst[2]:.1f}%)")
            
            # Compare my model to best/worst
            my_model_idx = summary_df[summary_df['Detector'] == 'My Model'].index
            if len(my_model_idx) > 0:
                try:
                    my_acc = float(summary_df.loc[my_model_idx[0], 'Paraphrased Accuracy'].replace('%', ''))
                    if my_acc >= best[2] * 0.95:  # Within 5% of best
                        print(f"• My model performs competitively on paraphrased text, achieving {my_acc:.1f}% accuracy.")
                    else:
                        print(f"• My model ({my_acc:.1f}%) lags behind the best detector ({best[2]:.1f}%) on paraphrased text.")
                except:
                    pass
else:
    print("• Insufficient data for interpretation.")

# STEP 12: Add detailed analysis
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

# Calculate overall accuracy for each detector
print("\nOverall Accuracy (all data):")

# My model
if 'my_model_pred' in df.columns:
    try:
        overall_acc = accuracy_score(df['true_label'], df['my_model_pred'])
        print(f"  My Model: {overall_acc:.2%}")
    except Exception as e:
        print(f"  My Model: Error - {e}")

# Commercial detectors
for detector in COLUMN_MAPPINGS['detectors']:
    if detector in df.columns and f'{detector}_binary' in df.columns:
        try:
            overall_acc = accuracy_score(df['true_label'], df[f'{detector}_binary'])
            print(f"  {detector}: {overall_acc:.2%}")
        except Exception as e:
            print(f"  {detector}: Error - {e}")

# Calculate paraphrasing impact for each detector
print("\nImpact of Paraphrasing (Accuracy Drop):")
for _, row in summary_df.iterrows():
    if row['Accuracy Drop'] != "N/A":
        print(f"  {row['Detector']}: {row['Accuracy Drop']} drop when paraphrased")

# STEP 13: Save results
print("\n[10] Saving results...")
os.makedirs('results', exist_ok=True)
summary_df.to_csv('results/paraphrase_analysis_summary.csv', index=False)

# Save detailed results
detailed_results = {
    'Detector': [],
    'Overall_Accuracy': [],
    'Paraphrased_Accuracy': [],
    'Non_Paraphrased_Accuracy': [],
    'Accuracy_Drop': []
}

for _, row in summary_df.iterrows():
    detailed_results['Detector'].append(row['Detector'])
    detailed_results['Paraphrased_Accuracy'].append(row['Paraphrased Accuracy'])
    detailed_results['Non_Paraphrased_Accuracy'].append(row['Non-Paraphrased Accuracy'])
    detailed_results['Accuracy_Drop'].append(row['Accuracy Drop'])
    
    # Add overall accuracy
    detector = row['Detector']
    if detector == 'My Model' and 'my_model_pred' in df.columns:
        try:
            overall_acc = accuracy_score(df['true_label'], df['my_model_pred'])
            detailed_results['Overall_Accuracy'].append(f"{overall_acc:.2%}")
        except:
            detailed_results['Overall_Accuracy'].append("N/A")
    elif detector in COLUMN_MAPPINGS['detectors']:
        if f'{detector}_binary' in df.columns:
            try:
                overall_acc = accuracy_score(df['true_label'], df[f'{detector}_binary'])
                detailed_results['Overall_Accuracy'].append(f"{overall_acc:.2%}")
            except:
                detailed_results['Overall_Accuracy'].append("N/A")
        else:
            detailed_results['Overall_Accuracy'].append("N/A")
    else:
        detailed_results['Overall_Accuracy'].append("N/A")

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv('results/paraphrase_detailed_analysis.csv', index=False)

print("✓ Summary saved to results/paraphrase_analysis_summary.csv")
print("✓ Detailed results saved to results/paraphrase_detailed_analysis.csv")

print("\n" + "="*60)
print("✅ Analysis complete!")
print("="*60)