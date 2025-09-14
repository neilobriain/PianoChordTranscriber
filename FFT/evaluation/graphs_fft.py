"""
Utility script to plot per-class accuracy bar chart,
accuracy distribution histogram, and confusion matrix map
after evaluate_fft.py has been run.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET = 'aPTD'

# File paths
class_acc_csv = f"{DATASET} FFT class_accuracy.csv"
confusion_csv = f"{DATASET} FFT confusion_pairs.csv"

# Load data
acc_df = pd.read_csv(class_acc_csv)
conf_df = pd.read_csv(confusion_csv)

# 1. Per-class accuracy bar chart
plt.figure(figsize=(14,6))
sns.barplot(x="Chord", y="Accuracy (%)", data=acc_df.sort_values("Accuracy (%)", ascending=False), palette="viridis")
plt.xticks(rotation=90)
plt.title("Per-Chord Accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel("Chord")
plt.tight_layout()
plt.show()

# 2. Accuracy distribution histogram
plt.figure(figsize=(8,5))
sns.histplot(acc_df["Accuracy (%)"], bins=10, kde=False, color="skyblue")
plt.title("Distribution of Chord Accuracies")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Chords")
plt.tight_layout()
plt.show()

# 3. Confusion matrix heatmap
conf_matrix = conf_df.set_index("True/Predicted")
plt.figure(figsize=(12,10))
sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="magma", cbar_kws={'label': 'Count'})
plt.title("Chord Confusion Matrix")
plt.ylabel("True Chord")
plt.xlabel("Predicted Chord")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()