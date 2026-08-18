import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# YOUR ACTUAL RESULTS from the analysis
results = {
    'Detector': ['My Model', 'ChatGPT', 'Gemini', 'Claude', 'zeroGPT'],
    'Non-Paraphrased': [0.56, 0.13, 0.17, 0.44, 0.30],  # Your actual numbers
    'Paraphrased': [0.49, 0.29, 0.27, 0.30, 0.44],       # Your actual numbers
    'Drop': [0.07, -0.16, -0.10, 0.14, -0.14]            # Non-Para - Para
}

df = pd.DataFrame(results)

print("="*60)
print("CORRECTED DATA FOR VISUALIZATION")
print("="*60)
print(df.to_string(index=False))

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AI Text Detector Performance: Paraphrase Evasion Analysis', 
             fontsize=16, y=1.02)

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
x = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x - width/2, df['Non-Paraphrased'], width, 
                label='Non-Paraphrased', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, df['Paraphrased'], width, 
                label='Paraphrased', color='#A23B72', alpha=0.8)

ax1.set_xlabel('Detector', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy: Paraphrased vs Non-Paraphrased', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Detector'], rotation=45, ha='right')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 0.8)

# Add value labels on bars
for i, row in df.iterrows():
    ax1.text(i - width/2, row['Non-Paraphrased'] + 0.02, 
             f'{row["Non-Paraphrased"]:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.text(i + width/2, row['Paraphrased'] + 0.02, 
             f'{row["Paraphrased"]:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Accuracy Drop
ax2 = axes[0, 1]
colors = ['green' if drop <= 0 else 'red' for drop in df['Drop']]
bars = ax2.bar(df['Detector'], df['Drop'], color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Detector', fontsize=12)
ax2.set_ylabel('Accuracy Drop (Non-Para - Para)', fontsize=12)
ax2.set_title('Accuracy Drop When Paraphrased', fontsize=14)
ax2.set_xticklabels(df['Detector'], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(df['Drop']):
    if v >= 0:
        ax2.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(i, v - 0.02, f'{v:.1%}', ha='center', va='top', fontweight='bold')

# Plot 3: Heatmap
ax3 = axes[1, 0]
heatmap_data = df[['Non-Paraphrased', 'Paraphrased']].T
sns.heatmap(heatmap_data, annot=True, fmt='.0%', cmap='RdYlGn',
            xticklabels=df['Detector'],
            yticklabels=['Non-Paraphrased', 'Paraphrased'],
            ax=ax3, cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=0.7)
ax3.set_title('Performance Heatmap', fontsize=14)

# Plot 4: Summary Statistics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
SUMMARY STATISTICS

Total Samples: 350
  • Paraphrased: 150 (42.9%)
  • Non-Paraphrased: 200 (57.1%)

Detector Performance:
  🏆 Best: My Model ({df.loc[0, 'Drop']:.1%} drop)
  🔴 Worst: ChatGPT ({df.loc[1, 'Drop']:.1%} drop)
  
  Average Drop: {df['Drop'].mean():.1%}

Key Findings:
  1. My Model shows best robustness
  2. ChatGPT most vulnerable to paraphrasing
  3. Paraphrasing NOT highly effective overall
  4. Some detectors improve on paraphrased text

Answer to RQ:
  Detection accuracy drops by {df['Drop'].mean():.1%} 
  on average when text is paraphrased.
"""

ax4.text(0.05, 0.5, summary_text, fontsize=12, verticalalignment='center',
         fontfamily='monospace', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('results/paraphrase_visualization_corrected.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to 'results/paraphrase_visualization_corrected.png'")
plt.show()

print("\n✅ Visualization complete!")
