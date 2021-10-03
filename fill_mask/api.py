from typing import List, Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
from model import get_model, get_tokenizer


app = FastAPI()


class FillMaskRequest(BaseModel):
    text: str


class FillMaskResponse(BaseModel):
    answers: List[Dict]
    # Each filled sentence is represented as a dict
    # Example of a dict for an answer:
    # {'sequence': '<s> This is a fill mask pipeline', 'score': 0.156278535723686, 'token': 3944, 'token_str' : 'mask'}


@app.post("/fill-mask", response_model=FillMaskResponse)
def fill_mask(
    request: FillMaskRequest,
    model: AutoModelForMaskedLM = Depends(get_model),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
):
    text = request.text

    fill_mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    result = fill_mask_pipeline(text)

    # FastAPI cannot json encode numpy.float32
    for entity_dict in result:
        entity_dict["score"] = float(entity_dict["score"])

    return FillMaskResponse(answers=result)
