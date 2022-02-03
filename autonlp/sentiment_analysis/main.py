from fastapi import APIRouter
from pydantic import BaseModel
from autonlp.sentiment_analysis import infer

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


@router.post("/validate/")
async def validate(request: SentimentRequest):
    return infer(request.text)
