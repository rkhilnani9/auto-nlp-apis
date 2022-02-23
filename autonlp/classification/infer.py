import re
import torch
import numpy as np


from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from autonlp.classification.model import get_model, get_tokenizer
from autonlp import config

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text, pretrained):
    model = get_model()
    tokenizer = get_tokenizer()
    input_text = re.sub(r"[^\w\s]", "", input_text).lower()

    if pretrained == "true":
        sentiment_classifier = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )
        preds = sentiment_classifier(input_text)[0]
    else:
        sentiment_classifier = AutoModelForSequenceClassification.from_pretrained(config.CLS_MODEL_SAVE_PATH)
        tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)
        tokens = tokenizer(input_text, return_tensors="pt")
        logits = sentiment_classifier(**tokens)
        pred_prob = np.max(torch.sigmoid(logits["logits"]).detach().cpu().numpy())
        label = "POSITIVE" if pred_prob > 0.5 else "NEGATIVE"
        preds = {"label" : label, "score" : float(pred_prob)}


    return preds


