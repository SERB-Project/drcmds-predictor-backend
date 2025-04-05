from fastapi import APIRouter, HTTPException
from app.schemas.pythogenicityPredictionReq import PathogenicityInput
from app.services.pathogenicityClassification.preprocess import preprocess_input
from app.services.pathogenicityClassification.predict import predict_pathogenicity

router = APIRouter()

@router.post("/predictPathogenicity")
async def predict_pathogenicity_endpoint(input_data: PathogenicityInput):
    try:
        # Preprocess input
        df = preprocess_input(input_data.spdi, input_data.consequences)
        
        # Predict using model
        result = predict_pathogenicity(df)
        
        return {
            "spdi": input_data.spdi,
            "consequences": input_data.consequences,
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
