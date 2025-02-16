import os
import json
import csv

# Specify the folder containing the JSON files and the output CSV file name
folder_path = 'data'
output_csv = 'data/combined3.csv'

# Open the CSV file for writing
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(["filename", "model", "utterance", "intent"])
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    content = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")
                    continue
                
                model = content.get("model", "")
                data_entries = content.get("data", [])
                
                # Write a row for each data entry in the JSON file
                for entry in data_entries:
                    utterance = entry.get("utterance", "")
                    intent = entry.get("intent", "")
                    writer.writerow([filename, model, utterance, intent])
                    
print(f"CSV file '{output_csv}' has been created successfully.")
