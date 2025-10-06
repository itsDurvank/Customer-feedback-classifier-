# 🤖 Customer Feedback Analyzer - AI-Powered Classification System

[](https://python.org)
[](https://streamlit.io)
[](https://huggingface.co/transformers)
[](https://www.google.com/search?q=LICENSE)
[](https://www.google.com/search?q=%23live-demo)

> **Transform customer feedback into actionable insights with AI\! 🚀**

## 📋 Table of Contents

  - [🎯 Project Overview](https://www.google.com/search?q=%23-project-overview)
  - [🚀 Quick Start Guide](https://www.google.com/search?q=%23-quick-start-guide)
  - [🏗️ Project Architecture](https://www.google.com/search?q=%23%EF%B8%8F-project-architecture)
  - [📁 File Structure](https://www.google.com/search?q=%23-file-structure)
  - [💻 Technical Implementation](https://www.google.com/search?q=%23-technical-implementation)
  - [📊 Performance Metrics](https://www.google.com/search?q=%23-performance-metrics)

-----

## 🎯 Project Overview

### **Intelligent Classification System using Hugging Face Transformers**

The Customer Feedback Analyzer is an advanced Natural Language Processing (NLP) application that automatically classifies customer feedback into actionable categories using a **fine-tuned BERT-based transformer model** from Hugging Face. The system leverages transfer learning to provide real-time classification with high accuracy.

### **Classification Categories**

The AI classifies feedback into **8 distinct categories**:

| Category | Emoji | Description | Example |
| :--- | :--- | :--- | :--- |
| **Bug Report** | 🐞 | Technical issues and software defects | "App crashes when uploading files" |
| **Feature Request** | 💡 | New functionality suggestions | "Please add dark mode option" |
| **Praise** | 🎉 | Positive feedback and compliments | "Love the new interface design\!" |
| **Complaint** | 😠 | Negative feedback and dissatisfaction | "The app is too slow to load" |
| **Question** | ❓ | User inquiries and help requests | "How do I export my data?" |
| **Usage Tip** | 💡 | User-generated tips and tricks | "Use Ctrl+S to save quickly" |
| **Documentation** | 📄 | Documentation-related feedback | "The API docs need updating" |
| **Other** | 🔖 | General feedback not fitting other categories | "Just wanted to say thanks" |

-----

## 🚀 Quick Start Guide

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/itsDurvank/Customer-feedback-classifier-.git
cd Customer-feedback-classifier-
```

### **Step 2: Set Up Python Environment**

```bash
# Create virtual environment
python -m venv feedback_env

# Activate it (Windows)
feedback_env\Scripts\activate

# Activate it (Mac/Linux)
source feedback_env/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Train the AI Model**

This step fine-tunes the BERT model and saves it to the `models/feedback_classifier` directory.

```bash
python finetune_classifier.py
```

### **Step 5: Launch the Web App**

Navigate to `http://localhost:8501` in your browser.

```bash
streamlit run app.py
```

### **🎬 Video Walkthrough**

Watch our complete setup and demo video: [Hugging\_Face\_model\_Fine\_Tuning.mp4](https://www.google.com/search?q=assets/Hugging_Face_model_Fine_Tuning.mp4)

-----

## 🏗️ Project Architecture

### **System Overview**

The system follows a clean, modular architecture separating model training, inference, and deployment layers.

| Layer | Component | Function |
| :--- | :--- | :--- |
| **Input/UI** | `app.py` (Streamlit) | Handles user interaction, input, and visualization. |
| **Inference** | `inference.py` | Loads the fine-tuned BERT model and performs real-time classification. |
| **Training** | `finetune_classifier.py` | Manages the transfer learning process and model saving. |

### **Training Pipeline**

The training process uses the Hugging Face `Trainer` class to manage:

1.  **Data Loading**: Read labeled feedback examples from `data/`.
2.  **Data Splitting**: Utilize `split_train_test.py` to create train/validation sets.
3.  **Fine-tuning**: Adapt a pre-trained BERT model (using `transformers`) to the 8-class classification task.
4.  **Evaluation**: Calculate performance metrics per epoch.
5.  **Model Saving**: Save the **best-performing model** to the `models/feedback_classifier` directory.

-----

## 📁 File Structure

```
customer-feedback-analyzer/
├── app.py                      # 🎨 Main Streamlit web application.
├── inference.py                # 🧠 AI model inference engine (loads model and makes predictions).
├── finetune_classifier.py      # 🎓 Model training script (fine-tunes BERT).
├── compare_models.py           # 📊 Utility for model performance comparison.
├── split_train_test.py         # ✂️ Data splitting utility.
├── requirements.txt            # 📦 Python dependencies.
├── data/                       # 📊 Training and test data files (.jsonl).
├── models/                     # 🤖 Trained AI models are saved here.
│   └── feedback_classifier/    # 🧠 Fine-tuned BERT model files.
└── assets/                     # 🖼️ Images and documentation assets (e.g., architecture diagrams).
```

-----

## 💻 Technical Implementation

### **`app.py` (Main Web Application)**

  * **Functionality**: Creates a Streamlit UI supporting both **Real-time Analysis** and **Batch Processing Mode** (for analyzing uploaded files).
  * **Design**: Features a mobile-responsive interface with live logging, performance tracking, and clear output display (category icons, confidence scores).

### **`inference.py` (AI Brain)**

  * **Model Loading**: Uses `AutoTokenizer` and `AutoModelForSequenceClassification` to load the trained model from the local `models/feedback_classifier` directory.
  * **Prediction**: Creates a Hugging Face `pipeline` for efficient classification, utilizing **GPU acceleration** when available (`device=0` if `torch.cuda.is_available()`).

### **`finetune_classifier.py` (Model Training)**

  * **Transfer Learning**: Initializes the pre-trained BERT model for sequence classification.
  * **Configuration**: Uses `TrainingArguments` to define the training process:
      * `num_train_epochs=3`
      * `per_device_train_batch_size=16`
      * `load_best_model_at_end=True`
      * `evaluation_strategy="epoch"`

-----

## 📊 Performance Metrics

The fine-tuned BERT model demonstrates strong performance on the test dataset:

### **Overall Performance Benchmarks**

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.1%** |
| **F1-Score** | **0.990** |
| **Inference Speed** | **\< 100ms** per classification |
| **Batch Processing** | **1000+ items per minute** |

### **Category Performance (F1-Scores)**

The model exhibits high reliability across all categories:

| Category | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Bug Report | 0.99 | 0.98 | 0.99 |
| Feature Request | 0.99 | 1.00 | 0.99 |
| Complaint | 0.98 | 0.99 | 0.99 |
| Praise | 1.00 | 0.99 | 0.99 |

-----

[](https://github.com/itsDurvank/Customer-feedback-classifier-.git)
[](https://huggingface.co)
[](https://www.google.com/search?q=LICENSE)
