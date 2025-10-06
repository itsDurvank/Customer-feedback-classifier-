import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import os

from transformers import TrainerCallback

class RealTimeFeedbackCallback(TrainerCallback):
    def __init__(self, tokenizer, id2label, label2id, train_dataset):
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = label2id
        self.train_dataset = train_dataset

    def on_evaluate(self, args, state, control, **kwargs):
        # Get predictions and labels
        predictions = kwargs.get('metrics', {}).get('eval_predictions', None)
        labels = kwargs.get('metrics', {}).get('eval_labels', None)
        if predictions is not None and labels is not None:
            preds = np.argmax(predictions, axis=-1)
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    text = self.train_dataset[i]['text'] if 'text' in self.train_dataset[i] else None
                    print(f"\nMisclassified sample: '{text}'")
                    print(f"Predicted: {self.id2label[pred]}, Actual: {self.id2label[label]}")
                    feedback = input("Is the prediction correct? (y/n): ")
                    if feedback.lower() == 'n':
                        correct_label = input(f"Enter correct label from {list(self.label2id.keys())}: ")
                        if correct_label in self.label2id:
                            # Add corrected sample to training set
                            self.train_dataset.append({'text': text, 'label': correct_label})
                            print("Sample added to training set for next epoch.")
                        else:
                            print("Invalid label. Skipping.")

labels = [
    "bug", "feature_request", "praise", "complaint", "question",
    "usage_tip", "documentation", "other"
]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

def preprocess(example):
    enc = tokenizer(example["text"], truncation=True, max_length=128)
    enc["label"] = label2id[example["label"]]
    return enc

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train_classifier_model():
    # Load data
    data = []
    with open("data/feedback_classify_train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    if len(data) < 10:
        raise ValueError("Training data is too small! Please add more labeled examples to feedback_classify_train.jsonl.")


    dataset = Dataset.from_list(data)
    model_ckpt = "bert-base-cased"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    dataset = dataset.map(preprocess)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir="models/feedback_classifier",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="logs",
        learning_rate=1e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2
    )

    from transformers import EarlyStoppingCallback
    feedback_callback = RealTimeFeedbackCallback(tokenizer, id2label, label2id, train_dataset)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), feedback_callback]
    )

    for epoch in range(int(args.num_train_epochs)):
        print(f"\nEpoch {epoch+1}/{int(args.num_train_epochs)}")
        trainer.train()
        # If new feedback samples were added, update train_dataset
        if len(train_dataset) > len(dataset["train"]):
            print("Retraining with new feedback samples...")
            trainer.train_dataset = train_dataset

    os.makedirs("models/feedback_classifier", exist_ok=True)
    trainer.save_model("models/feedback_classifier")
    tokenizer.save_pretrained("models/feedback_classifier")
    print("Model and tokenizer saved to models/feedback_classifier")

if __name__ == "__main__":
    train_classifier_model()