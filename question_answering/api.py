import re

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from model import get_model, get_tokenizer


app = FastAPI()


class QuestionAnsweringRequest(BaseModel):
    context: str
    question: str


class QuestionAnsweringResponse(BaseModel):
    answer: str
    start_index: int
    end_index: int
    score: float


@app.post("/question-answering", response_model=QuestionAnsweringResponse)
def answer_question(
    request: QuestionAnsweringRequest,
    model: AutoModelForQuestionAnswering = Depends(get_model),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
):
    context = request.context.lower()
    question = request.question.lower()

    question_answering_pipeline = pipeline(
        "question-answering", model=model, tokenizer=tokenizer
    )

    result = question_answering_pipeline(question, context)
    print(result)

    return QuestionAnsweringResponse(
        answer=result["answer"],
        start_index=result["start"],
        end_index=result["end"],
        score=result["score"],
    )
