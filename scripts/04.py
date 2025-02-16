from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import time


ATP = "02"

LABELS = ['greeting', 'insult', 'create_deployment_plan', 'modify_deployment_plan', 'related_question', 'ask_plan_details', 'unrelated_question']
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}



# === Step 1: Load the Fine-Tuned Model ===
MODEL_PATH = f"{ATP}/fine_tuned_small_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set model to evaluation mode

while True:
    # Ask for user input
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    t = time.time()
    # Tokenize the user input
    inputs = tokenizer([user_input], padding=True, truncation=True, return_tensors="pt")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to label predictions
    predicted_label = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]

    # Convert ID to actual label
    predicted_intent = id2label[predicted_label]

    # Display result
    print(f"Text: {user_input} → Predicted Intent: {predicted_intent} {predicted_label}")
    print(f"Time taken: {time.time() - t:.3f} seconds")
