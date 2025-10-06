import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report

# Paths to test set and models
test_file = "data/feedback_classify_test.jsonl"  # Your test set file
baseline_ckpt = "bert-base-cased"                # Baseline: Pretrained BERT (no fine-tuning)
your_model_ckpt = "models/feedback_classifier"    # Your fine-tuned model

# Helper: Predict class indices for a list of texts
def predict(model_dir, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()
        preds.append(pred)
    # If model has id2label, return it; otherwise, return empty dict
    id2label = getattr(model.config, "id2label", None)
    if id2label is None:
        id2label = {}
    return preds, id2label

# Load test set
texts, labels = [], []
with open(test_file, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        texts.append(ex["text"])
        labels.append(ex["label"])


# Use the same label order as in training for correct mapping
labels_order = [
    "bug", "feature_request", "praise", "complaint", "question",
    "usage_tip", "documentation", "other"
]
label2id = {label: i for i, label in enumerate(labels_order)}
y_true = [label2id[label] for label in labels]

# Predict with baseline model (bert-base-cased, not fine-tuned)
preds_base, id2label_base = predict(baseline_ckpt, texts)

# Predict with your fine-tuned model
preds_yours, id2label_yours = predict(your_model_ckpt, texts)

print("\n--- Baseline Model (bert-base-cased, not fine-tuned) ---")
print(classification_report(y_true, preds_base, target_names=labels_order))

print("\n--- Your Fine-tuned Model ---")
print(classification_report(y_true, preds_yours, target_names=labels_order))