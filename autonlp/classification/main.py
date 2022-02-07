from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse

router = APIRouter()


class ClassificationRequest(BaseModel):
    text: str


@router.post("/validate/")
async def validate(request: ClassificationRequest):
    from autonlp.classification.infer import infer
    return infer(request.text)


@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    from autonlp.classification.train import train_model
    train_model(dataframe)
    return {"Message": "Model training has started!"}


@router.post('/predict/')
async def predict(dataframe: UploadFile = File(...)):
    from autonlp.classification.predict import predict
    return JSONResponse(predict(dataframe))
