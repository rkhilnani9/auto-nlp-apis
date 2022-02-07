from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel


router = APIRouter()


class NerRequest(BaseModel):
    text: str


@router.post('/validate/')
def predict_ner_tags(request: NerRequest):
    from autonlp.ner.infer import infer
    return infer(request.text)

@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    from autonlp.ner.train import train_model
    train_model(dataframe)
    return {"Message": "Model training has started!"}


@router.post('/predict/')
async def predict(dataframe: UploadFile = File(...)):
    from autonlp.ner.predict import predict
    return predict(dataframe)
