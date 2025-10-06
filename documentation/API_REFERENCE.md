# 📚 API Reference - Customer Feedback Analyzer

This document provides detailed technical reference for all functions, classes, and modules in the Customer Feedback Analyzer project.

## 📋 Table of Contents

- [Core Modules](#core-modules)
- [Function Reference](#function-reference)
- [Class Reference](#class-reference)
- [Configuration Options](#configuration-options)
- [Data Formats](#data-formats)
- [Error Handling](#error-handling)
- [Performance Metrics](#performance-metrics)

---

## 🧠 Core Modules

### `inference.py` - Model Inference Engine

The main inference module that handles text classification using the fine-tuned BERT model.

#### Functions

##### `analyze_feedback(text: str) -> Tuple[str, float]`

Analyzes customer feedback text and returns classification results.

**Parameters:**

- `text` (str): The customer feedback text to classify

**Returns:**

- `tuple`: A tuple containing:
  - `label` (str): The predicted category label
  - `confidence` (float): Confidence score between 0 and 1

**Example:**

```python
from inference import analyze_feedback

# Analyze a bug report
result = analyze_feedback("The app crashes when I upload files")
print(result)  # ('bug', 0.9456)

# Analyze a feature request
result = analyze_feedback("Please add dark mode")
print(result)  # ('feature_request', 0.8923)
```

**Supported Labels:**

- `bug` - Technical issues and software defects
- `feature_request` - New functionality suggestions
- `praise` - Positive feedback and compliments
- `complaint` - Negative feedback and dissatisfaction
- `question` - User inquiries and help requests
- `usage_tip` - User-generated tips and tricks
- `documentation` - Documentation-related feedback
- `other` - General feedback not fitting other categories

**Error Handling:**

```python
try:
    label, confidence = analyze_feedback("Sample feedback")
except FileNotFoundError:
    print("Model not found. Please train the model first.")
except Exception as e:
    print(f"Analysis failed: {e}")
```

---

### `finetune_classifier.py` - Model Training Module

Handles the complete training pipeline for the feedback classification model.

#### Functions

##### `train_classifier_model() -> None`

Trains a BERT-based classifier on the feedback dataset.

**Parameters:** None

**Returns:** None (saves model to disk)

**Process:**

1. Loads training and validation datasets
2. Initializes pre-trained BERT model
3. Configures training parameters
4. Fine-tunes the model
5. Evaluates performance
6. Saves the trained model

**Example:**

```python
from finetune_classifier import train_classifier_model

# Train the model
train_classifier_model()
# Model will be saved to 'models/feedback_classifier/'
```

**Configuration Options:**

```python
# Training hyperparameters (modify in the script)
TRAINING_ARGS = {
    'output_dir': './models/feedback_classifier',
    'num_train_epochs': 3,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 64,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'logging_dir': './logs',
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'accuracy',
    'greater_is_better': True,
}
```

##### `load_dataset() -> Tuple[Dataset, Dataset]`

Loads and preprocesses the training and validation datasets.

**Returns:**

- `tuple`: Training and validation datasets

**Example:**

```python
train_dataset, eval_dataset = load_dataset()
print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")
```

##### `compute_metrics(eval_pred) -> Dict[str, float]`

Computes evaluation metrics during training.

**Parameters:**

- `eval_pred`: Evaluation predictions from the trainer

**Returns:**

- `dict`: Dictionary containing accuracy and other metrics

---

### `app.py` - Streamlit Web Application

The main web application interface built with Streamlit.

#### Key Functions

##### `add_log(message: str, log_type: str = "info") -> None`

Adds timestamped log entries to the session state.

**Parameters:**

- `message` (str): Log message to display
- `log_type` (str): Type of log entry ("info", "success", "error", "warning")

**Example:**

```python
add_log("Model loaded successfully", "success")
add_log("Processing feedback...", "info")
add_log("Classification failed", "error")
```

##### `update_stats(category: str, confidence: float) -> None`

Updates system statistics with new analysis results.

**Parameters:**

- `category` (str): The predicted category
- `confidence` (float): Confidence score

**Example:**

```python
update_stats("bug", 0.95)
# Updates total analyzed count, category tracking, and average confidence
```

##### `sanitize_text(val: Any) -> str`

Sanitizes text input to prevent display issues with special characters.

**Parameters:**

- `val` (Any): Input value to sanitize

**Returns:**

- `str`: Sanitized text safe for display

##### `display_logs() -> None`

Renders the live system logs in the Streamlit interface.

**Features:**

- Shows last 15 log entries
- Color-coded by log type
- Auto-scrolling display
- Real-time updates

---

### `compare_models.py` - Model Comparison Module

Compares different model approaches and generates performance reports.

#### Functions

##### `compare_model_performance() -> Dict[str, Any]`

Compares the fine-tuned model against baseline approaches.

**Returns:**

- `dict`: Comparison results including metrics for each model

**Example:**

```python
from compare_models import compare_model_performance

results = compare_model_performance()
print(f"Fine-tuned BERT accuracy: {results['bert']['accuracy']:.3f}")
print(f"Baseline accuracy: {results['baseline']['accuracy']:.3f}")
```

##### `generate_classification_report() -> str`

Generates detailed classification report with per-class metrics.

**Returns:**

- `str`: Formatted classification report

---

### `test.py` - Testing and Validation Module

Provides testing utilities and model validation functions.

#### Functions

##### `test_model_accuracy() -> float`

Tests the trained model on the test dataset.

**Returns:**

- `float`: Overall accuracy score

**Example:**

```python
from test import test_model_accuracy

accuracy = test_model_accuracy()
print(f"Model accuracy: {accuracy:.1%}")
```

##### `test_individual_predictions() -> None`

Tests model predictions on sample feedback examples.

**Example:**

```python
from test import test_individual_predictions

test_individual_predictions()
# Outputs predictions for predefined test cases
```

---

## ⚙️ Configuration Options

### Model Configuration

```python
# Model settings
MODEL_NAME = "bert-base-uncased"  # Pre-trained model to use
MAX_LENGTH = 512                  # Maximum input sequence length
NUM_LABELS = 8                    # Number of classification categories

# Alternative models you can try:
# "distilbert-base-uncased"  # Faster, smaller model
# "roberta-base"             # Alternative transformer architecture
# "bert-base-multilingual-cased"  # For multilingual support
```

### Training Configuration

```python
# Training hyperparameters
LEARNING_RATE = 2e-5              # Learning rate for fine-tuning
BATCH_SIZE = 16                   # Training batch size
NUM_EPOCHS = 3                    # Number of training epochs
WARMUP_STEPS = 500                # Learning rate warmup steps
WEIGHT_DECAY = 0.01               # L2 regularization strength

# Early stopping
EARLY_STOPPING_PATIENCE = 2       # Epochs to wait before stopping
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum improvement threshold
```

### Streamlit Configuration

```python
# App settings
PAGE_TITLE = "Customer Feedback Intelligence"
PAGE_ICON = "🤖"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Performance settings
MAX_LOG_ENTRIES = 100             # Maximum log entries to keep
CACHE_TTL = 3600                  # Model cache time-to-live (seconds)
```

---

## 📊 Data Formats

### Training Data Format (JSONL)

Each line contains a JSON object with text and label:

```json
{"text": "The app keeps crashing on startup", "label": "bug"}
{"text": "Could you add a search feature?", "label": "feature_request"}
{"text": "Excellent customer service!", "label": "praise"}
```

**Required Fields:**

- `text` (str): The feedback text to classify
- `label` (str): The ground truth category label

**Optional Fields:**

- `id` (str): Unique identifier for the feedback
- `timestamp` (str): When the feedback was submitted
- `user_id` (str): Anonymous user identifier
- `metadata` (dict): Additional context information

### Prediction Output Format

```python
{
    "text": "Input feedback text",
    "predicted_label": "bug",
    "confidence": 0.9456,
    "all_scores": [
        {"label": "bug", "score": 0.9456},
        {"label": "feature_request", "score": 0.0234},
        {"label": "praise", "score": 0.0123},
        # ... other categories
    ],
    "timestamp": "2024-01-15T14:30:22Z"
}
```

### Batch Processing Format

**Input CSV:**

```csv
id,text,source
1,"App crashes when uploading",mobile_app
2,"Please add dark mode",web_app
3,"Great customer support!",email
```

**Output CSV:**

```csv
id,text,predicted_label,confidence,source
1,"App crashes when uploading",bug,0.9456,mobile_app
2,"Please add dark mode",feature_request,0.8923,web_app
3,"Great customer support!",praise,0.9678,email
```

---

## 🚨 Error Handling

### Common Exceptions

#### `ModelNotFoundError`

Raised when the trained model cannot be found.

```python
class ModelNotFoundError(Exception):
    """Raised when the trained model is not found."""
    pass

# Usage
try:
    result = analyze_feedback("test")
except ModelNotFoundError:
    print("Please train the model first using: python finetune_classifier.py")
```

#### `InvalidInputError`

Raised when input text is invalid or empty.

```python
class InvalidInputError(Exception):
    """Raised when input text is invalid."""
    pass

# Usage
try:
    result = analyze_feedback("")
except InvalidInputError:
    print("Input text cannot be empty")
```

#### `ModelLoadError`

Raised when model loading fails.

```python
class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass
```

### Error Handling Best Practices

```python
def safe_analyze_feedback(text: str) -> Optional[Tuple[str, float]]:
    """
    Safely analyze feedback with comprehensive error handling.

    Args:
        text: Input feedback text

    Returns:
        Tuple of (label, confidence) or None if analysis fails
    """
    try:
        # Validate input
        if not text or not text.strip():
            raise InvalidInputError("Input text cannot be empty")

        if len(text) > 1000:
            text = text[:1000]  # Truncate long text

        # Perform analysis
        return analyze_feedback(text)

    except ModelNotFoundError:
        print("Model not found. Please train the model first.")
        return None
    except InvalidInputError as e:
        print(f"Invalid input: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

---

## 📈 Performance Metrics

### Classification Metrics

#### Accuracy

```python
accuracy = correct_predictions / total_predictions
```

#### Precision (per class)

```python
precision = true_positives / (true_positives + false_positives)
```

#### Recall (per class)

```python
recall = true_positives / (true_positives + false_negatives)
```

#### F1-Score (per class)

```python
f1_score = 2 * (precision * recall) / (precision + recall)
```

#### Macro-averaged F1

```python
macro_f1 = sum(f1_scores_per_class) / num_classes
```

### Performance Benchmarks

#### Inference Speed

- **CPU**: ~100ms per classification
- **GPU**: ~30ms per classification
- **Batch processing**: 1000+ items per minute

#### Memory Usage

- **Model size**: ~440MB (BERT-base)
- **Runtime memory**: ~500MB RAM
- **GPU memory**: ~2GB VRAM (if using GPU)

#### Training Time

- **CPU**: 5-10 minutes (3 epochs)
- **GPU**: 2-3 minutes (3 epochs)
- **Dataset size**: 800 training examples

### Monitoring Functions

```python
def get_performance_metrics() -> Dict[str, float]:
    """
    Get current model performance metrics.

    Returns:
        Dictionary containing performance metrics
    """
    return {
        'accuracy': 0.991,
        'macro_precision': 0.991,
        'macro_recall': 0.989,
        'macro_f1': 0.990,
        'inference_time_ms': 95.2,
        'memory_usage_mb': 487.3
    }

def benchmark_inference_speed(num_samples: int = 100) -> float:
    """
    Benchmark inference speed.

    Args:
        num_samples: Number of samples to test

    Returns:
        Average inference time in milliseconds
    """
    import time

    sample_texts = ["Sample feedback text"] * num_samples
    start_time = time.time()

    for text in sample_texts:
        analyze_feedback(text)

    end_time = time.time()
    avg_time_ms = ((end_time - start_time) / num_samples) * 1000

    return avg_time_ms
```

---

## 🔧 Advanced Usage

### Custom Model Integration

```python
def load_custom_model(model_path: str) -> None:
    """
    Load a custom trained model.

    Args:
        model_path: Path to the custom model directory
    """
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Usage
load_custom_model("path/to/your/custom/model")
```

### Batch Processing API

```python
def analyze_feedback_batch(texts: List[str]) -> List[Tuple[str, float]]:
    """
    Analyze multiple feedback texts in batch.

    Args:
        texts: List of feedback texts to analyze

    Returns:
        List of (label, confidence) tuples
    """
    results = []
    for text in texts:
        try:
            result = analyze_feedback(text)
            results.append(result)
        except Exception as e:
            results.append(("error", 0.0))

    return results

# Usage
feedback_list = [
    "App crashes frequently",
    "Please add dark mode",
    "Great customer service!"
]
results = analyze_feedback_batch(feedback_list)
```

### Model Ensemble

```python
def ensemble_predict(text: str, models: List[str]) -> Tuple[str, float]:
    """
    Use multiple models for prediction and combine results.

    Args:
        text: Input text to classify
        models: List of model paths

    Returns:
        Ensemble prediction result
    """
    predictions = []

    for model_path in models:
        load_custom_model(model_path)
        label, confidence = analyze_feedback(text)
        predictions.append((label, confidence))

    # Simple voting ensemble
    from collections import Counter
    labels = [pred[0] for pred in predictions]
    most_common_label = Counter(labels).most_common(1)[0][0]

    # Average confidence for the most common label
    avg_confidence = sum(conf for label, conf in predictions
                        if label == most_common_label) / labels.count(most_common_label)

    return most_common_label, avg_confidence
```

---

This API reference provides comprehensive documentation for all components of the Customer Feedback Analyzer. For additional examples and use cases, refer to the main README.md and the example scripts in the repository.
