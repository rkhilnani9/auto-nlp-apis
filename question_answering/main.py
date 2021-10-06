from fastapi import APIRouter
from pydantic import BaseModel
from question_answering.infer import infer

router = APIRouter()


class QARequest(BaseModel):
    text: str
    context: str


@router.post("/validate/")
def answer_question(request: QARequest):
    return infer(request.text, request.context)
