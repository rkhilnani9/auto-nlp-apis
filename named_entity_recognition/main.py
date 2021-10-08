from fastapi import APIRouter
from pydantic import BaseModel
from named_entity_recognition.infer import infer

router = APIRouter()


class NerRequest(BaseModel):
    text: str


@router.post("/validate/")
def predict_ner_tags(request: NerRequest):
    return infer(request.text)
