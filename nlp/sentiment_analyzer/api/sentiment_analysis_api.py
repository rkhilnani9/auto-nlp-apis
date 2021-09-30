import re

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from models.get_sentiment_analysis_model import get_model, get_tokenizer


app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    confidence: float


@app.post("/sentiment-analyzer", response_model=SentimentResponse)
def predict_sentiment(
    request: SentimentRequest,
    model: AutoModelForSequenceClassification = Depends(get_model),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
):
    input_text = request.text

    input_text = input_text.lower()
    input_text = re.sub(r"[^\w\s]", "", input_text)
    sentiment_classifier = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )

    result = sentiment_classifier(input_text)[0]

    return SentimentResponse(label=result["label"], confidence=result["score"])
