from fastapi import APIRouter
from pydantic import BaseModel
from fill_mask.infer import infer


router = APIRouter()


class FillMaskRequest(BaseModel):
    text: str


@router.post("/validate/")
async def fill_mask(request: FillMaskRequest):
    return infer(request.text)
