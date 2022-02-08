from fastapi import APIRouter
from pydantic import BaseModel
from autonlp.summarization.infer import infer

router = APIRouter()


class SummarizationRequest(BaseModel):
    text: str


@router.post("/infer/")
async def infer(request: SummarizationRequest):
    return infer(request.text)
