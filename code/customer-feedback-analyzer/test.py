import json
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter

# Paths to your data files
train_path = "data/feedback_classify_train.jsonl"
test_path = "data/feedback_classify_test.jsonl"

# Load train data
train_texts, train_labels = [], []
with open(train_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            train_texts.append(obj["text"])
            train_labels.append(obj["label"])

# Load test data
test_texts, test_labels = [], []
with open(test_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            test_texts.append(obj["text"])
            test_labels.append(obj["label"])

# Map labels to ids
labels = ["bug", "feature_request", "praise", "complaint", "question", "usage_tip", "documentation", "other"]
label2id = {label: i for i, label in enumerate(labels)}
train_label_ids = [label2id[l] for l in train_labels]
test_label_ids = [label2id[l] for l in test_labels]

# Print class distributions
print("Train class distribution:", Counter(train_labels))
print("Test class distribution:", Counter(test_labels))

# Load model and tokenizer
model_path = "models/feedback_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def get_preds(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return preds

train_preds = get_preds(train_texts)
print("Training Performance:")
print(classification_report(train_label_ids, train_preds, target_names=labels))

test_preds = get_preds(test_texts)
print("Test Performance:")
print(classification_report(test_label_ids, test_preds, target_names=labels))