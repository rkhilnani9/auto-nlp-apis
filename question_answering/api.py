import re

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from model import get_model, get_tokenizer


app = FastAPI()


class QuestionAnsweringRequest(BaseModel):
    text: str


class QuestionAnsweringResponse(BaseModel):
    label: str
    confidence: float


@app.post("/question-answering", response_model=QuestionAnsweringRequest)
def answer_question(
    request: QuestionAnsweringRequest,
    model: AutoModelForQuestionAnswering = Depends(get_model),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
):
    input_text = request.text

    input_text = input_text.lower()

    question_answering_pipeline = pipeline(
        "question-answering", model=model, tokenizer=tokenizer
    )

    result = question_answering_pipeline(input_text)
    print(result)

    return QuestionAnsweringResponse(label=result["label"], confidence=result["score"])
