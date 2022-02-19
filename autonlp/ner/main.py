from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from autonlp.ner.infer import infer
from autonlp.ner.train import train_model


router = APIRouter()


class NerRequest(BaseModel):
    text: str


@router.post("/infer/{pretrained}")
async def validate(request: NerRequest, pretrained: str = "true"):
    return infer(request.text, pretrained)


@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    train_model(dataframe)
    return {"Message": "Model training has started!"}
