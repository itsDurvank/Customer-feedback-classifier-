# Code Snippets or Lines Explanation

---

# Expanded Code Snippets and Explanations

## 1. Streamlit Application (`app.py`)

### File Overview

This file creates the interactive web interface for the feedback analyzer. It manages user input, session state, logging, batch analysis, and displays analytics.

### Key Imports

```python
import streamlit as st
import os
import time
import json
from datetime import datetime
from inference import analyze_feedback
import plotly.express as px
import pandas as pd
```

- `streamlit`: Main UI framework.
- `inference.analyze_feedback`: Core function for prediction.
- `plotly`, `pandas`: For analytics and charts.

### Session State Initialization

```python
if "logs" not in st.session_state:
    st.session_state.logs = []
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "system_stats" not in st.session_state:
    st.session_state.system_stats = {
        "total_analyzed": 0,
        "model_accuracy": 0.99,
        "avg_confidence": 0.0,
        "categories_detected": set()
    }
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
```

- Ensures all session variables are initialized for logging, stats, and input control.

### Logging and Stats Functions

```python
def add_log(message, log_type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.logs.append({"message": log_entry, "type": log_type, "time": timestamp})
    if len(st.session_state.logs) > 100:
        st.session_state.logs = st.session_state.logs[-100:]

def update_stats(category, confidence):
    st.session_state.system_stats["total_analyzed"] += 1
    st.session_state.system_stats["categories_detected"].add(category)
    current_avg = st.session_state.system_stats["avg_confidence"]
    total = st.session_state.system_stats["total_analyzed"]
    st.session_state.system_stats["avg_confidence"] = ((current_avg * (total - 1)) + confidence) / total
```

- `add_log`: Adds timestamped log entries, keeps only the last 100.
- `update_stats`: Maintains rolling average confidence and tracks detected categories.

### Main Analysis Logic

```python
category, confidence = analyze_feedback(feedback_text)
update_stats(category, confidence)
st.session_state.analysis_history.append({
    "text": feedback_text,
    "category": category,
    "confidence": confidence,
    "timestamp": datetime.now()
})
```

- Calls the model, updates stats, and appends to history for analytics.

### Batch Mode

```python
if batch_mode:
    feedback_list = [line.strip() for line in batch_feedbacks.splitlines() if line.strip()]
    for i, fb in enumerate(feedback_list):
        category, confidence = analyze_feedback(fb)
        update_stats(category, confidence)
        st.session_state.analysis_history.append({
            "text": fb,
            "category": category,
            "confidence": confidence,
            "timestamp": datetime.now()
        })
```

- Supports analyzing multiple feedbacks at once, updating stats/history for each.

### Analytics and Visualization

```python
history_df = pd.DataFrame(st.session_state.analysis_history)
category_counts = history_df['category'].value_counts()
fig = px.pie(values=category_counts.values, names=category_counts.index, title="Feedback Category Distribution")
st.plotly_chart(fig, use_container_width=True)
```

- Shows category distribution as a pie chart.

---

## 2. Inference Engine (`inference.py`)

### File Overview

Loads the trained model and tokenizer, runs predictions using Hugging Face pipeline, and returns the top label and confidence score.

### Core Function

```python
def analyze_feedback(text):
    model_dir = "models/feedback_classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    nlp = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1,
    )
    result = nlp(text)[0]
    top = max(result, key=lambda x: x["score"])
    return top["label"], top["score"]
```

- Loads model/tokenizer from disk.
- Uses GPU if available for faster inference.
- Returns the label with the highest score.

### Explanation

- **Pipeline**: Handles tokenization, batching, and device management.
- **Device selection**: `device=0` for CUDA, `-1` for CPU.
- **Return**: Human-readable label and confidence (float).

---

## 3. Model Training (`finetune_classifier.py`)

### File Overview

Loads data, maps labels, tokenizes, sets up training arguments, and fine-tunes BERT using Hugging Face Trainer.

### Label Mapping

```python
labels = ["bug", "feature_request", "praise", "complaint", "question", "usage_tip", "documentation", "other"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
```

- Ensures consistent mapping for training and inference.

### Data Preprocessing

```python
def preprocess(example):
    enc = tokenizer(example["text"], truncation=True, max_length=128)
    enc["label"] = label2id[example["label"]]
    return enc
```

- Tokenizes text and attaches numeric label.

### Training Arguments

```python
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
```

- Conservative learning rate for BERT stability.
- Saves best model by F1 score.

### Trainer and Callbacks

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), feedback_callback]
)
```

- Early stopping prevents overfitting.
- Feedback callback enables human-in-the-loop error analysis.

### Saving Model

```python
trainer.save_model("models/feedback_classifier")
tokenizer.save_pretrained("models/feedback_classifier")
```

- Persists model and tokenizer for later inference.

---

## 4. Model Comparison (`compare_models.py`)

### File Overview

Compares baseline (pretrained BERT) and fine-tuned model on test set, prints classification reports.

### Prediction Helper

```python
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
    id2label = getattr(model.config, "id2label", None)
    if id2label is None:
        id2label = {}
    return preds, id2label
```

- Runs inference for each text, returns predicted class indices.

### Evaluation

```python
print("--- Baseline Model (bert-base-cased, not fine-tuned) ---")
print(classification_report(y_true, preds_base, target_names=labels_order))
print("--- Your Fine-tuned Model ---")
print(classification_report(y_true, preds_yours, target_names=labels_order))
```

- Prints per-class and overall metrics for both models.

---

## 5. Data Split (`split_train_test.py`)

### File Overview

Splits labeled data into train/test sets with stratification, writes to disk, and prints class distributions.

### Stratified Split

```python
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=test_ratio, stratify=labels, random_state=42
)
```

- Ensures label balance in both sets.

### Writing Files

```python
with open(train_file, "w", encoding="utf-8") as f:
    for text, label in zip(train_texts, train_labels):
        f.write(json.dumps({"text": text, "label": label}) + "\n")
```

- Saves train set in JSONL format.

### Class Distribution

```python
from collections import Counter
print("Train class distribution:", Counter(train_labels))
print("Test class distribution:", Counter(test_labels))
```

- Prints counts for quick verification.

---

## 6. Model Testing (`test.py`)

### File Overview

Evaluates trained model on train and test sets, prints classification reports.

### Data Loading

```python
with open(train_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            train_texts.append(obj["text"])
            train_labels.append(obj["label"])
```

- Loads labeled data from disk.

### Prediction

```python
def get_preds(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return preds
```

- Runs batch inference for efficiency.

### Reporting

```python
print("Training Performance:")
print(classification_report(train_label_ids, train_preds, target_names=labels))
print("Test Performance:")
print(classification_report(test_label_ids, test_preds, target_names=labels))
```

- Prints metrics for both train and test sets.

---

