from fastapi import APIRouter

router = APIRouter()

# Import endpoints and include them in the main router
from app.api.endpoints import predict

# example of how to import and include endpoints in the main router
# router.include_router(predict.router, prefix="/predict", tags=["Model Prediction"])
