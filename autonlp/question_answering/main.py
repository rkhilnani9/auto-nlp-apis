from fastapi import APIRouter
from pydantic import BaseModel
from autonlp.question_answering.infer import infer

router = APIRouter()


class QARequest(BaseModel):
    text: str
    context: str


@router.post("/infer/")
def answer_question(request: QARequest):
    return infer(request.text, request.context)
