from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from autonlp.classification.infer import infer
from autonlp.classification.train import train_model
from autonlp.classification.predict import predict

router = APIRouter()


class ClassificationRequest(BaseModel):
    text: str


@router.post("/infer/")
async def validate(request: ClassificationRequest):
    return infer(request.text)


@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    train_model(dataframe)
    return {"Message": "Model training has started!"}


@router.post('/predict/')
async def predict(dataframe: UploadFile = File(...)):
    return JSONResponse(predict(dataframe))
