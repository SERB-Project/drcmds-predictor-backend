from fastapi import APIRouter
from app.api.endpoints import sarsVariantPrediction
from app.api.endpoints import pathogenicityPrediction
from app.api.endpoints import viralDiseasePrediction
from app.api.endpoints import spliceSitePrediction

router = APIRouter()

router.include_router(sarsVariantPrediction.router, prefix="/sars-variants", tags=["SARS-CoV2 Variants Classification and Mutation Analysis"])
router.include_router(pathogenicityPrediction.router, prefix="/pathogenicity", tags=["Pathogenicity Classification"])
router.include_router(viralDiseasePrediction.router, prefix="/viral-disease", tags=["Viral Disease Prediction"])
router.include_router(spliceSitePrediction.router, prefix="/splice-site-prediction", tags=["Splice Site Prediction"])