import csv
import json

# Config 
CSV_FILE = "data/pubmedqa_test_clean.csv"          
JSON_FILE = "data/ori_pqal.json"   
OUTPUT_FILE = "data/actual_testing_dataset.json"  

# Load test questions from CSV
test_questions = set()
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = row['sent1'].strip()
        test_questions.add(question)

print(f"Loaded {len(test_questions)} unique test questions.")

# Load full JSON dataset
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# Filter JSON data based on matching questions
filtered_data = {}
for pmid, entry in full_data.items():
    if entry.get("QUESTION", "").strip() in test_questions:
        filtered_data[pmid] = entry

print(f"Filtered dataset contains {len(filtered_data)} entries.")

# Save to new JSON
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Saved filtered dataset to {OUTPUT_FILE}")
