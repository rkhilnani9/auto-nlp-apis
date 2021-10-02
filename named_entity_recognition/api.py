import re

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from model import get_model, get_tokenizer


app = FastAPI()


class NerRequest(BaseModel):
    text: str


class NerResponse(BaseModel):
    label: str
    confidence: float


@app.post("/named-entity-recognizer", response_model=NerResponse)
def predict_ner_tags(
    request: NerRequest,
    model: AutoModelForTokenClassification = Depends(get_model),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
):
    input_text = request.text

    input_text = input_text.lower()

    named_entity_recognizer = pipeline(
        "ner", model=model, tokenizer=tokenizer
    )

    result = named_entity_recognizer(input_text)
    print(result)

    return NerResponse(label=result["label"], confidence=result["score"])
