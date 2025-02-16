import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

ATP = "03"

# Load dataset
df_test = pd.read_csv(f"{ATP}/input.csv", usecols=['utterance', 'intent'], encoding="windows-1252")
LABELS = ['greeting', 'insult', 'create_deployment_plan', 'modify_deployment_plan', 'related_question', 'ask_plan_details', 'unrelated_question']
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

# Save the datasets to CSV
dataset_val_temp = dataset_val.rename_columns({"text": "utterance", "label": "intent"}).to_pandas()
dataset_val_temp['intent'] = dataset_val_temp['intent'].map(id2label)
dataset_val_temp.to_csv(f"{ATP}/val_dataset.csv", index=False)

# Load a small tokenizer and model (Replace Phi-3 with a smaller model)
MODEL_NAME = "distilbert-base-uncased"  # Alternative: "bert-tiny", "albert-base-v2", "tiny-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="longest", truncation=True)  # Adjusted padding for efficiency

dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)

dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Training arguments (Reduce batch size if needed)
training_args = TrainingArguments(
    output_dir=f"{ATP}/results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,  # Reduce for lower memory usage
    per_device_eval_batch_size=4,
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
trainer.save_model(f"{ATP}/fine_tuned_small_model")
tokenizer.save_pretrained(f"{ATP}/fine_tuned_small_model")

# Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
