from fastapi import APIRouter
from app.api.endpoints import sarsVariantPrediction
from app.api.endpoints import pathogenicityPrediction


router = APIRouter()

router.include_router(sarsVariantPrediction.router, prefix="/sars-variants", tags=["SARS-CoV2 Variants Classification and Mutation Analysis"])
router.include_router(pathogenicityPrediction.router, prefix="/pathogenicity", tags=["Pathogenicity Classification"])