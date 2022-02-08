from fastapi import APIRouter
from pydantic import BaseModel
from autonlp.fill_mask.infer import infer


router = APIRouter()


class FillMaskRequest(BaseModel):
    text: str


@router.post("/infer/")
async def fill_mask(request: FillMaskRequest):
    return infer(request.text)
