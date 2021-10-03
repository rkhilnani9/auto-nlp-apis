from typing import List, Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from model import get_model, get_tokenizer


app = FastAPI()


class NerRequest(BaseModel):
    text: str


class NerResponse(BaseModel):
    entities: List[Dict]
    # Each tagged entity is represented as a dict
    # Example of a dict for an entity:
    # {entity_group': 'ORG', 'score': 0.6889763, 'word': 'face', 'start': 8, 'end': 12}


@app.post("/named-entity-recognizer", response_model=NerResponse)
def predict_ner_tags(
    request: NerRequest,
    model: AutoModelForTokenClassification = Depends(get_model),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
):
    input_text = request.text.lower()

    named_entity_recognizer = pipeline(
        "ner", model=model, tokenizer=tokenizer, grouped_entities=True
    )

    result = named_entity_recognizer(input_text)

    # FastAPI cannot json encode numpy.float32
    for entity_dict in result:
        entity_dict["score"] = float(entity_dict["score"])

    return NerResponse(entities=result)
