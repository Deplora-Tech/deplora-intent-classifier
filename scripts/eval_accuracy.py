import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load the Predicted Responses from JSON ---
# Replace 'responses.json' with the path to your JSON file
with open("02\\eval\\deepseek\\res.json", "r") as f:
    predictions = json.load(f)
    
# predictions is expected to be a dict with:
# { "utterance": "predicted_intent", ... }

# --- Step 2: Load the Correct Intents from CSV ---
# Replace 'correct_intents.csv' with the path to your CSV file
df_correct = pd.read_csv("02/test.csv", encoding="windows-1252")

# The CSV is expected to have fields: 'utterance' and 'intent'

# --- Step 3: Merge Predictions with Correct Labels ---
# Map the predicted intents using the utterance as key
df_correct['predicted_intent'] = df_correct['utterance'].map(predictions)

# Warn if any utterances did not receive a prediction
if df_correct['predicted_intent'].isnull().any():
    missing = df_correct[df_correct['predicted_intent'].isnull()]
    print("Warning: The following utterances did not receive a prediction:")
    print(missing['utterance'].tolist())
    # Optionally, drop these rows:
    df_correct = df_correct.dropna(subset=['predicted_intent'])

# --- Step 4: Compute Evaluation Metrics ---
true_labels = df_correct['intent'].tolist()
pred_labels = df_correct['predicted_intent'].tolist()

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"✅ Accuracy: {accuracy:.4f}\n")

# Generate a detailed classification report (includes precision, recall, and F1-score)
report = classification_report(true_labels, pred_labels)
print("🔹 Classification Report:")
print(report)

# Compute confusion matrix
labels = sorted(list(set(true_labels) | set(pred_labels)))
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# --- Step 5: Visualize the Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- Step 6: Output Wrong Predictions ---
wrong_predictions = df_correct[df_correct['intent'] != df_correct['predicted_intent']]
print("❌ Wrong Predictions:")
print(wrong_predictions[['utterance', 'intent', 'predicted_intent']])
