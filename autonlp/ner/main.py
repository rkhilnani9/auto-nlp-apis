from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from autonlp.ner.infer import infer
from autonlp.ner.train import train_model
# from autonlp.ner.predict import predict


router = APIRouter()


class NerRequest(BaseModel):
    text: str


@router.post('/infer/')
def infer(request: NerRequest):
    return infer(request.text)

@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    train_model(dataframe)
    return {"Message": "Model training has started!"}


# @router.post('/predict/')
# async def predict(dataframe: UploadFile = File(...)):
#     return predict(dataframe)
