from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

labels = [
    "bug", "feature_request", "praise", "complaint", "question",
    "usage_tip", "documentation", "other"
]

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