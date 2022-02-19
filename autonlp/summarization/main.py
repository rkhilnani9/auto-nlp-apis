from fastapi import APIRouter
from pydantic import BaseModel
from autonlp.summarization.infer import infer

router = APIRouter()


class SummarizationRequest(BaseModel):
    text: str


@router.post("/infer/{pretrained}")
async def validate(request: SummarizationRequest, pretrained: str = "true"):
    return infer(request.text, pretrained)
