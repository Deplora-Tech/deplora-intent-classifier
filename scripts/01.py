import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Load dataset
df_test = pd.read_csv("Bitext_Sample_Customer_Service_Testing_Dataset.csv", usecols=['utterance', 'intent'])
LABELS = df_test['intent'].unique()
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

df_test['label'] = df_test['intent'].map(label2id)

# Split into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_test['utterance'].tolist(), df_test['label'].tolist(), test_size=0.2, random_state=42
)

# Convert to Hugging Face dataset
def preprocess_data(texts, labels):
    return Dataset.from_dict({'text': texts, 'label': labels})

dataset_train = preprocess_data(train_texts, train_labels)
dataset_val = preprocess_data(val_texts, val_labels)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/Phi-3-mini-4k-instruct", num_labels=len(LABELS))

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)

dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_phi3")
tokenizer.save_pretrained("./fine_tuned_phi3")

# Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)