from fastapi import APIRouter
from pydantic import BaseModel
from sentiment_analysis.infer import infer

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


@router.post("/validate/")
async def validate(request: SentimentRequest):
    return infer(request.text)
