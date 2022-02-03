import re
from transformers import pipeline
from autonlp.sentiment_analysis import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text):
    input_text = re.sub(r"[^\w\s]", "", input_text).lower()

    sentiment_classifier = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )

    return sentiment_classifier(input_text)[0]
