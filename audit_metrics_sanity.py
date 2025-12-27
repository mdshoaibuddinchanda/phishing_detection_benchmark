import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Example: 5 samples
predictions = np.array([1, 0, 1, 1, 0])  # Model predictions
true_labels = np.array([1, 0, 0, 1, 1])  # True labels

print("Predictions:", predictions)
print("True labels:", true_labels)
print()

# Manual computation
tp = np.sum((predictions == 1) & (true_labels == 1))
tn = np.sum((predictions == 0) & (true_labels == 0))
fp = np.sum((predictions == 1) & (true_labels == 0))
fn = np.sum((predictions == 0) & (true_labels == 1))

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print()

# Accuracy: (TP + TN) / (TP + TN + FP + FN)
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy (manual): {accuracy_manual:.4f}")
print(f"Accuracy (sklearn): {accuracy_score(true_labels, predictions):.4f}")
print()

# Precision: TP / (TP + FP)
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"Precision (manual): {precision_manual:.4f}")
print(f"Precision (sklearn): {precision_score(true_labels, predictions):.4f}")
print()

# Recall: TP / (TP + FN)
recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Recall (manual): {recall_manual:.4f}")
print(f"Recall (sklearn): {recall_score(true_labels, predictions):.4f}")
print()

# F1: 2 * (Precision * Recall) / (Precision + Recall)
if (precision_manual + recall_manual) > 0:
    f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual)
else:
    f1_manual = 0
print(f"F1-Score (manual): {f1_manual:.4f}")
print(f"F1-Score (sklearn): {f1_score(true_labels, predictions):.4f}")
print()

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)
