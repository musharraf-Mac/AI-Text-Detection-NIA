# obj5_complete.py - AI Detection Paraphrase Analysis

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_model_and_vectorizer():
    """Load the trained baseline model and vectorizer"""
    try:
        with open('models/baseline_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Model and vectorizer loaded successfully")
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def load_benchmark_data():
    """Load the paraphrase benchmark dataset"""
    try:
        df = pd.read_csv('data/processed/paraphrase_benchmark_cleaned.csv')
        print(f"✓ Benchmark data loaded: {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError as e:
        print(f"✗ Error loading benchmark data: {e}")
        return None

def run_my_model_predictions(df, model, vectorizer):
    """Run predictions using your trained model"""
    if model is None or vectorizer is None:
        return None
    
    try:
        # Transform text using the same vectorizer
        X_test = vectorizer.transform(df['text'])
        
        # Get predictions (assuming binary classification: 0=Human, 1=AI)
        predictions = model.predict(X_test)
        
        # Add to dataframe
        df['my_model_pred'] = predictions
        print(f"✓ My model predictions completed")
        return df
    except Exception as e:
        print(f"✗ Error in predictions: {e}")
        return None

def calculate_accuracy_by_group(df, pred_col, true_col='author'):
    """Calculate accuracy for paraphrased vs non-paraphrased text"""
    results = {}
    
    # Convert author to binary if needed (assuming 'AI' vs 'Human')
    if true_col in df.columns:
        # Map author to binary (adjust based on your actual values)
        if df[true_col].dtype == 'object':
            # Check what values exist
            unique_authors = df[true_col].unique()
            print(f"  Author values: {unique_authors}")
            # Assuming 'AI' or 'Machine' vs 'Human' - adjust as needed
            if 'AI' in unique_authors or 'Machine' in unique_authors:
                df['true_label'] = df[true_col].apply(
                    lambda x: 1 if x in ['AI', 'Machine', 'GPT', 'Claude', 'Bard'] else 0
                )
            else:
                # If binary already, just copy
                df['true_label'] = df[true_col]
    
    # For each group
    for group_name, group_data in df.groupby('paraphrased'):
        if pred_col in group_data.columns:
            accuracy = accuracy_score(group_data['true_label'], group_data[pred_col])
            results[group_name] = {
                'accuracy': accuracy,
                'count': len(group_data)
            }
    
    return results

def analyze_commercial_detectors(df):
    """Analyze performance of the 4 commercial detectors"""
    detector_cols = ['ChatGPT', 'Bard', 'Claude', 'zeroGPT']
    detector_results = {}
    
    for detector in detector_cols:
        if detector in df.columns:
            # Map detector values to binary (0=Human, 1=AI)
            # Assuming detector columns contain 'AI'/'Human' or 0/1
            detector_values = df[detector].unique()
            print(f"  {detector} values: {detector_values}")
            
            # Convert to binary if needed
            if df[detector].dtype == 'object':
                df[f'{detector}_binary'] = df[detector].apply(
                    lambda x: 1 if str(x).upper() in ['AI', 'MACHINE', 'GPT', '1'] else 0
                )
            else:
                df[f'{detector}_binary'] = df[detector]
            
            # Calculate accuracy for each group
            detector_results[detector] = calculate_accuracy_by_group(
                df, f'{detector}_binary'
            )
    
    return detector_results

def create_summary_table(df, my_model_results, detector_results):
    """Create a comprehensive summary table"""
    
    # Initialize the summary dictionary
    summary = {
        'Detector': [],
        'Paraphrased Accuracy': [],
        'Non-Paraphrased Accuracy': [],
        'Accuracy Drop': [],
        'Paraphrased Count': [],
        'Non-Paraphrased Count': []
    }
    
    # Add my model results
    if my_model_results:
        summary['Detector'].append('My Model')
        summary['Paraphrased Accuracy'].append(my_model_results.get(True, {}).get('accuracy', np.nan))
        summary['Non-Paraphrased Accuracy'].append(my_model_results.get(False, {}).get('accuracy', np.nan))
        summary['Paraphrased Count'].append(my_model_results.get(True, {}).get('count', 0))
        summary['Non-Paraphrased Count'].append(my_model_results.get(False, {}).get('count', 0))
    
    # Add commercial detectors
    for detector, results in detector_results.items():
        summary['Detector'].append(detector)
        summary['Paraphrased Accuracy'].append(results.get(True, {}).get('accuracy', np.nan))
        summary['Non-Paraphrased Accuracy'].append(results.get(False, {}).get('accuracy', np.nan))
        summary['Paraphrased Count'].append(results.get(True, {}).get('count', 0))
        summary['Non-Paraphrased Count'].append(results.get(False, {}).get('count', 0))
    
    # Calculate accuracy drop
    summary['Accuracy Drop'] = [
        (non_para - para) if not (np.isnan(para) or np.isnan(non_para)) else np.nan
        for para, non_para in zip(summary['Paraphrased Accuracy'], summary['Non-Paraphrased Accuracy'])
    ]
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary)
    
    # Format percentage columns
    for col in ['Paraphrased Accuracy', 'Non-Paraphrased Accuracy', 'Accuracy Drop']:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "N/A")
    
    return summary_df

def main():
    """Main execution function"""
    print("="*60)
    print("AI Detection Paraphrase Analysis")
    print("="*60)
    
    # Step 1: Load model and data
    print("\n[1] Loading model and data...")
    model, vectorizer = load_model_and_vectorizer()
    df = load_benchmark_data()
    
    if df is None:
        return
    
    # Step 2: Run my model predictions
    print("\n[2] Running my model predictions...")
    df = run_my_model_predictions(df, model, vectorizer)
    
    # Step 3: Calculate my model accuracy by group
    print("\n[3] Calculating my model accuracy...")
    if 'my_model_pred' in df.columns:
        my_results = calculate_accuracy_by_group(df, 'my_model_pred')
        print(f"  My model - Paraphrased: {my_results.get(True, {}).get('accuracy', 0):.2%} "
              f"({my_results.get(True, {}).get('count', 0)} samples)")
        print(f"  My model - Non-paraphrased: {my_results.get(False, {}).get('accuracy', 0):.2%} "
              f"({my_results.get(False, {}).get('count', 0)} samples)")
    
    # Step 4: Analyze commercial detectors
    print("\n[4] Analyzing commercial detectors...")
    detector_results = analyze_commercial_detectors(df)
    
    # Step 5: Create summary table
    print("\n[5] Creating summary table...")
    summary_df = create_summary_table(df, my_results if 'my_model_pred' in df.columns else None, detector_results)
    
    print("\n" + "="*60)
    print("SUMMARY TABLE: Accuracy on Paraphrased vs Non-Paraphrased Text")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Step 6: Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    # Calculate average drop
    drop_values = []
    for i, row in summary_df.iterrows():
        if row['Accuracy Drop'] != "N/A":
            drop = float(row['Accuracy Drop'].replace('%', ''))
            drop_values.append(drop)
    
    if drop_values:
        avg_drop = np.mean(drop_values)
        print(f"• On average, all detectors show an accuracy drop of {avg_drop:.1f}% when text has been paraphrased.")
        
        if avg_drop > 10:
            print(f"• Paraphrasing appears to be an effective evasion technique, reducing detection accuracy by {avg_drop:.1f}% across all tools.")
        else:
            print(f"• Paraphrasing has a modest effect ({avg_drop:.1f}% drop), suggesting current detectors are relatively robust to paraphrasing.")
        
        # Find best and worst performers
        best_performer = summary_df.loc[summary_df['Paraphrased Accuracy'] != "N/A", 'Paraphrased Accuracy'].max()
        worst_performer = summary_df.loc[summary_df['Paraphrased Accuracy'] != "N/A", 'Paraphrased Accuracy'].min()
        
        print(f"• Among all detectors, {summary_df[summary_df['Paraphrased Accuracy'] == best_performer]['Detector'].iloc[0]} performs best on paraphrased text ({best_performer}), "
              f"while {summary_df[summary_df['Paraphrased Accuracy'] == worst_performer]['Detector'].iloc[0]} performs worst ({worst_performer}).")
    
    # Save results
    print("\n[6] Saving results...")
    summary_df.to_csv('results/paraphrase_analysis_summary.csv', index=False)
    print("✓ Summary saved to results/paraphrase_analysis_summary.csv")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()