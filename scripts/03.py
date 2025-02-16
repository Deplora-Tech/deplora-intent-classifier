import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset

ATP = "02"

# === Step 1: Load the Fine-Tuned Model ===
MODEL_PATH = f"{ATP}/fine_tuned_small_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set model to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Step 2: Load Evaluation Dataset ===
# Replace with the actual dataset file used for validation
EVAL_DATA_PATH = f"{ATP}/test.csv"
df_test = pd.read_csv(EVAL_DATA_PATH, usecols=['utterance', 'intent'], encoding="windows-1252")

# Load label mapping (from fine-tuning)
LABELS = ['greeting', 'insult', 'create_deployment_plan', 'modify_deployment_plan', 'related_question', 'ask_plan_details', 'unrelated_question']
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# Map intents to numerical labels
df_test['label'] = df_test['intent'].map(label2id)

# Convert to Hugging Face dataset format
dataset_val = Dataset.from_dict({'text': df_test['utterance'].tolist(), 'label': df_test['label'].tolist()})

# === Step 3: Tokenize Evaluation Data ===
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

dataset_val = dataset_val.map(tokenize_function, batched=True)
dataset_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# === Step 4: Run Model on Evaluation Data ===
# Convert dataset into PyTorch tensors
dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=8, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Get model predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print(all_labels)
print("\n\n\n")
print(all_preds)
# === Step 5: Compute Metrics ===
# Compute Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Validation Accuracy: {accuracy:.4f}")

# Compute Precision, Recall, and F1-score
report = classification_report(all_labels, all_preds, target_names=[str(label) for label in LABELS])
print("\n🔹 Classification Report:")
print(report)

# Compute Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# === Step 6: Visualize Confusion Matrix ===
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
