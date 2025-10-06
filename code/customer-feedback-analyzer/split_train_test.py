import json
from sklearn.model_selection import train_test_split

input_file = "data/feedback_classify_train.jsonl"
train_file = "data/feedback_classify_train.jsonl"
test_file = "data/feedback_classify_test.jsonl"
test_ratio = 0.2  # 20% for test

# Load data
with open(input_file, encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

# Stratified split to maintain class balance
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=test_ratio, stratify=labels, random_state=42
)

# Write train set
with open(train_file, "w", encoding="utf-8") as f:
    for text, label in zip(train_texts, train_labels):
        f.write(json.dumps({"text": text, "label": label}) + "\n")

# Write test set
with open(test_file, "w", encoding="utf-8") as f:
    for text, label in zip(test_texts, test_labels):
        f.write(json.dumps({"text": text, "label": label}) + "\n")

# Print class counts for quick verification
from collections import Counter
print("Train class distribution:", Counter(train_labels))
print("Test class distribution:", Counter(test_labels))
print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")