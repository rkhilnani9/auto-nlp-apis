from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    from classification.train import train_model
    train_model(dataframe)
    return {"Message": "Model training has started!"}


@router.post('/predict/')
async def predict(dataframe: UploadFile = File(...)):
    from classification.infer import infer
    return JSONResponse(infer(dataframe))
