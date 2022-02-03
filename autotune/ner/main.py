import shutil
import io
from fastapi import APIRouter, File, UploadFile

router = APIRouter()


@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    from ner.train import train_model
    train_model(dataframe)
    return {"Message": "Model training has started!"}


@router.post('/predict/')
async def predict(dataframe: UploadFile = File(...)):
    from ner.infer import infer
    return infer(dataframe)
