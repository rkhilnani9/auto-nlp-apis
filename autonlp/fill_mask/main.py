from fastapi import APIRouter
from pydantic import BaseModel
from autonlp.fill_mask.infer import infer


router = APIRouter()


class FillMaskRequest(BaseModel):
    text: str


@router.post("/infer/{pretrained}")
async def validate(request: FillMaskRequest, pretrained: str = "true"):
    return infer(request.text, pretrained)
