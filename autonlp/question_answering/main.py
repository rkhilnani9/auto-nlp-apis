from fastapi import APIRouter
from pydantic import BaseModel
from autonlp.question_answering.infer import infer

router = APIRouter()


class QARequest(BaseModel):
    text: str
    context: str


@router.post("/infer/{pretrained}")
async def validate(request: QARequest, pretrained: str = "true"):
    return infer(request.text, request.context, pretrained)

