    # Relevant Examples

    ## Code Snippets

    ### Inference: Top label with confidence

    ```python
    from inference import analyze_feedback
    label, score = analyze_feedback("Please add dark mode")
    print(f"Predicted label: {label}, Confidence: {score:.2f}")
    ```

    ### Training: Start fine-tuning

    ```python
    from finetune_classifier import train_classifier_model
    train_classifier_model()
    # Model and tokenizer will be saved to models/feedback_classifier/
    ```

    ### Model Comparison: Baseline vs Fine-tuned

    ```python
    # Compare baseline and fine-tuned model performance
    from compare_models import compare_model_performance  # see API docs for structure
    # Or directly run compare_models.py to print classification reports
    ```

    ### Data Split: Stratified

    ```python
    # split_train_test.py
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    ```

    ### Batch Inference Example

    ```python
    feedbacks = [
        "The app crashes when uploading files",
        "Please add dark mode",
        "Great support team!"
    ]
    results = [analyze_feedback(fb) for fb in feedbacks]
    for fb, (label, score) in zip(feedbacks, results):
        print(f"Input: {fb}\nPredicted: {label}, Confidence: {score:.2f}\n")
    ```

    ### Error Handling Example

    ```python
    try:
        label, score = analyze_feedback("")
    except Exception as e:
        print("Error:", e)
    ```

    ---

    ## Practical Inputs & Outputs

    - **Input:** "The app crashes when uploading files"  
    **Output:** `bug`, Confidence: 0.97
    - **Input:** "Please add dark mode"  
    **Output:** `feature_request`, Confidence: 0.94
    - **Input:** "Great support team!"  
    **Output:** `praise`, Confidence: 0.99
    - **Input:** "Documentation is unclear"  
    **Output:** `documentation`, Confidence: 0.91
    - **Input:** "How do I reset my password?"  
    **Output:** `question`, Confidence: 0.93

    ---

    ## Mermaid Diagrams

    ### 1. End-to-End Workflow

    ```mermaid
    flowchart TD
        A[Raw Feedback Data] --> B[Data Preprocessing]
        B --> C[Tokenization]
        C --> D[Model Training (BERT Fine-tuning)]
        D --> E[Trained Model]
        E --> F[Model Evaluation]
        F --> G[Deployment]
        G --> H[Real-Time Inference]
        H --> I[Feedback Classification Output]
        style A fill:#e3f2fd,color:#333,stroke:#2196f3
        style B fill:#f1f8e9,color:#333,stroke:#43a047
        style C fill:#fffde7,color:#333,stroke:#fbc02d
        style D fill:#fce4ec,color:#333,stroke:#d81b60
        style E fill:#ede7f6,color:#333,stroke:#5e35b1
        style F fill:#e0f7fa,color:#333,stroke:#00bcd4
        style G fill:#f3e5f5,color:#333,stroke:#8e24aa
        style H fill:#e8f5e9,color:#333,stroke:#388e3c
        style I fill:#fff3e0,color:#333,stroke:#ffb300
    ```

    ### 2. System Architecture

    ```mermaid
    graph LR
        subgraph Data Layer
            TD[Training Data (JSONL)]
            TSD[Test Data (JSONL)]
        end
        subgraph Model Layer
            BM[BERT Base Model]
            FC[Fine-tuned Classifier]
            BM --> FC
        end
        subgraph Interface Layer
            SW[Streamlit Web App]
            UI[User Interface]
        end
        IP[Inference Pipeline]
        TD --> FC
        TSD --> FC
        FC --> IP
        IP --> SW
        SW --> UI
        style TD fill:#e1f5fe,color:#333,stroke:#039be5
        style TSD fill:#e1f5fe,color:#333,stroke:#039be5
        style BM fill:#f3e5f5,color:#333,stroke:#8e24aa
        style FC fill:#f3e5f5,color:#333,stroke:#8e24aa
        style SW fill:#e8f5e8,color:#333,stroke:#43a047
        style UI fill:#e8f5e8,color:#333,stroke:#43a047
        style IP fill:#fff3e0,color:#333,stroke:#ffb300
    ```

    ### 3. Data Processing Flow

    ```mermaid
    flowchart LR
        IT[User Input Text] --> TOK[Tokenization & Encoding]
        TOK --> BM[BERT Model Encoder]
        BM --> CH[Classification Head]
        CH --> SM[Softmax & Prediction]
        SM --> CR[Classification Result]
        style IT fill:#e3f2fd,color:#333,stroke:#2196f3
        style TOK fill:#f1f8e9,color:#333,stroke:#43a047
        style BM fill:#fce4ec,color:#333,stroke:#d81b60
        style CH fill:#fffde7,color:#333,stroke:#fbc02d
        style SM fill:#ede7f6,color:#333,stroke:#5e35b1
        style CR fill:#e8f5e9,color:#333,stroke:#388e3c
    ```

    ### 4. Training Pipeline

    ```mermaid
    flowchart TD
        RD[Raw Data] --> DP[Data Preprocessing]
        DP --> TS[Train/Test Split]
        TS --> TK[Tokenization]
        TK --> MT[Model Training]
        PM[Pre-trained BERT] --> MT
        MT --> ME[Model Evaluation]
        ME --> MS[Model Saving]
        MS --> VI[Validation]
        VI --> PT[Performance Testing]
        PT --> MD[Model Deployment]
        style RD fill:#ffebee,color:#333,stroke:#e53935
        style DP fill:#f3e5f5,color:#333,stroke:#8e24aa
        style TS fill:#e8f5e8,color:#333,stroke:#43a047
        style TK fill:#e1f5fe,color:#333,stroke:#039be5
        style MT fill:#fff3e0,color:#333,stroke:#ffb300
        style ME fill:#f1f8e9,color:#333,stroke:#43a047
        style MS fill:#fce4ec,color:#333,stroke:#d81b60
        style VI fill:#ede7f6,color:#333,stroke:#5e35b1
        style PT fill:#e0f7fa,color:#333,stroke:#00bcd4
        style MD fill:#f3e5f5,color:#333,stroke:#8e24aa
    ```

    ### 5. User Interaction Flow

    ```mermaid
    sequenceDiagram
        participant U as User
        participant UI as Streamlit Interface
        participant ML as Model Loader
        participant IP as Inference Pipeline
        participant M as BERT Model

        U->>UI: Enter feedback text
        U->>UI: Click "Analyze" button
        UI->>ML: Check model availability
        alt Model not trained
            ML->>UI: Display training prompt
            U->>UI: Click "Train Model"
            UI->>ML: Initiate training process
            ML->>UI: Training completed
            UI->>U: Show success message
        else Model available
            UI->>IP: Send text for classification
            IP->>M: Tokenize and encode text
            M->>IP: Return prediction scores
            IP->>UI: Format classification result
            UI->>U: Display category and confidence
        end
    ```

    ---

    ## More Practical Examples

    ### Feedbacks and Expected Outputs

    | Input                         | Expected Label  | Confidence |
    | ----------------------------- | --------------- | ---------- |
    | "App is slow to respond"      | complaint       | 0.92       |
    | "Add export to PDF"           | feature_request | 0.90       |
    | "Found a typo in docs"        | documentation   | 0.88       |
    | "How do I change my email?"   | question        | 0.95       |
    | "Love the new dashboard!"     | praise          | 0.98       |
    | "Tip: Use keyboard shortcuts" | usage_tip       | 0.89       |
    | "App crashes on login"        | bug             | 0.96       |
    | "Other feedback"              | other           | 0.85       |

    ---

    ---

    ## Advanced Usage Examples

    ### Custom Confidence Threshold

    ```python
    def classify_with_threshold(text, threshold=0.8):
        label, score = analyze_feedback(text)
        if score >= threshold:
            return label
        else:
            return "Low confidence: manual review needed"

    result = classify_with_threshold("App is slow to respond", threshold=0.95)
    print(result)
    ```

    ### Multi-label Extension (Concept)

    ```python
    # For multi-label tasks, adapt the model and output handling
    def analyze_multi_label(text):
        # Returns list of (label, score) pairs
        ... # Custom implementation
        return [("bug", 0.7), ("feature_request", 0.6)]
    ```

    ### Edge Case Handling

    ```python
    inputs = [None, "", "12345", "😊👍"]
    for inp in inputs:
        try:
            label, score = analyze_feedback(inp)
            print(f"Input: {inp} -> {label}, {score}")
        except Exception as e:
            print(f"Input: {inp} -> Error: {e}")
    ```

    ---

    ## Batch Processing & Output Visualization

    ### Batch Mode Example

    ```python
    import pandas as pd
    feedbacks = [
        "App is slow to respond",
        "Add export to PDF",
        "Found a typo in docs",
        "How do I change my email?",
        "Love the new dashboard!",
        "Tip: Use keyboard shortcuts",
        "App crashes on login",
        "Other feedback"
    ]
    results = [analyze_feedback(fb) for fb in feedbacks]
    df = pd.DataFrame({
        "Feedback": feedbacks,
        "Label": [r[0] for r in results],
        "Confidence": [r[1] for r in results]
    })
    print(df)
    ```

    ---

    ## Troubleshooting & Common Issues

    ### Model Not Found

    ```python
    try:
        label, score = analyze_feedback("Test feedback")
    except FileNotFoundError:
        print("Model not found. Please run train_classifier_model() first.")
    ```

    ### CUDA Out of Memory

    ```python
    import torch
    try:
        # Run inference on large batch
        ...
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Reduce batch size or switch to CPU mode.")
    ```

    ---

    ## Deployment Visualization

    ### 6. Production Deployment Architecture

    ```mermaid
    flowchart LR
        LB[Load Balancer] --> APP1[Streamlit App 1]
        LB --> APP2[Streamlit App 2]
        LB --> APP3[Streamlit App N]
        APP1 --> MS1[Model Server 1]
        APP2 --> MS2[Model Server 2]
        APP3 --> MC[Model Cache]
        MS1 --> DB[(Database)]
        MS2 --> FS[File Storage]
        APP1 --> LOGS[Log Storage]
        APP2 --> LOGS
        APP3 --> LOGS
        LOGS --> MON[Monitoring]
        MON --> ALERT[Alerting]
        MON --> DASH[Dashboard]
        style LB fill:#e3f2fd,color:#333,stroke:#2196f3
        style APP1 fill:#e8f5e8,color:#333,stroke:#43a047
        style APP2 fill:#e8f5e8,color:#333,stroke:#43a047
        style APP3 fill:#e8f5e8,color:#333,stroke:#43a047
        style MS1 fill:#fff3e0,color:#333,stroke:#ffb300
        style MS2 fill:#fff3e0,color:#333,stroke:#ffb300
        style MC fill:#f3e5f5,color:#333,stroke:#8e24aa
        style DB fill:#ede7f6,color:#333,stroke:#5e35b1
        style FS fill:#ede7f6,color:#333,stroke:#5e35b1
        style LOGS fill:#e0f7fa,color:#333,stroke:#00bcd4
        style MON fill:#f1f8e9,color:#333,stroke:#43a047
        style ALERT fill:#ffebee,color:#333,stroke:#e53935
        style DASH fill:#fffde7,color:#333,stroke:#fbc02d
    ```

    ### 7. Microservices Architecture (Advanced)

    ```mermaid
    graph TB
        subgraph "API Gateway"
            AG[API Gateway\nAuthentication & Routing]
        end
        subgraph "Core Services"
            FS[Feedback Service\nInput Processing]
            CS[Classification Service\nML Inference]
            AS[Analytics Service\nMetrics & Insights]
        end
        subgraph "Data Services"
            DS[Data Service\nCRUD Operations]
            MS[Model Service\nModel Management]
            CS_CACHE[Cache Service\nRedis/Memcached]
        end
        subgraph "External Services"
            DB[(PostgreSQL\nDatabase)]
            S3[S3 Storage\nModels & Data]
            ELK[ELK Stack\nLogging]
        end
        AG --> FS
        AG --> CS
        AG --> AS
        FS --> DS
        CS --> MS
        AS --> DS
        DS --> DB
        MS --> S3
        CS --> CS_CACHE
        FS --> ELK
        CS --> ELK
        AS --> ELK
        style AG fill:#e3f2fd,color:#333,stroke:#2196f3
        style FS fill:#e8f5e8,color:#333,stroke:#43a047
        style CS fill:#fff3e0,color:#333,stroke:#ffb300
        style AS fill:#f3e5f5,color:#333,stroke:#8e24aa
        style DS fill:#fce4ec,color:#333,stroke:#d81b60
        style MS fill:#f1f8e9,color:#333,stroke:#43a047
        style CS_CACHE fill:#ede7f6,color:#333,stroke:#5e35b1
        style DB fill:#ede7f6,color:#333,stroke:#5e35b1
        style S3 fill:#ede7f6,color:#333,stroke:#5e35b1
        style ELK fill:#e0f7fa,color:#333,stroke:#00bcd4
    ```

    ---

    ## References

    - See `DOCUMENT.md` for full architecture and workflow explanations.
    - See `API_REFERENCE.md` for function details and usage.
    - See `app.py`, `inference.py`, `finetune_classifier.py` for implementation.
