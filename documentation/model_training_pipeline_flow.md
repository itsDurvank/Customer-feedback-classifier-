```mermaid
stateDiagram-v2
    [*] --> RawFeedbackData: Start
    RawFeedbackData --> DataPreprocessing
    DataPreprocessing --> Tokenization
    Tokenization --> ModelTraining: "BERT Fine-tuning"
    ModelTraining --> TrainedModel
    TrainedModel --> ModelEvaluation
    ModelEvaluation --> Deployment
    Deployment --> RealTimeInference
    RealTimeInference --> FeedbackClassificationOutput
    FeedbackClassificationOutput --> [*]: End

    RawFeedbackData: Raw Feedback Data\ndata/feedback_classify_train.jsonl\ndata/feedback_classify_test.jsonl
    DataPreprocessing: Data Preprocessing\nsplit_train_test.py
    Tokenization: Tokenization\nfinetune_classifier.py
    ModelTraining: Model Training\nfinetune_classifier.py\nBERT Fine-tuning
    TrainedModel: Trained Model\nmodels/feedback_classifier/
    ModelEvaluation: Model Evaluation\ntest.py, compare_models.py
    Deployment: Deployment\napp.py, inference.py
    RealTimeInference: Real-time Inference\ninference.py, app.py
    FeedbackClassificationOutput: Feedback Classification Output\napp.py UI
```

**File Flow Explanation:**

- Raw feedback data is preprocessed and tokenized.
- The tokenized data is used to fine-tune the BERT model.
- The trained model is evaluated and then deployed.
- The deployed model performs real-time inference and outputs classified feedback.
