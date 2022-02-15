from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from autonlp.classification.infer import infer
from autonlp.classification.train import train_model

router = APIRouter()


class ClassificationRequest(BaseModel):
    text: str


@router.post("/infer/{pretrained}")
async def infer(request: ClassificationRequest, pretrained: str = "true"):
    return infer(request.text, pretrained)


@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    train_model(dataframe)
    return {"Message": "Model training has started!"}
