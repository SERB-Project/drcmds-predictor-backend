from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.services.viralDiseasePrediction.predict import predict_sequence, predict_file
from app.schemas.viralDiseasePrediction import SequenceInput
from app.services.viralDiseasePrediction.preprocess import pad_sequences, one_hot_encoding
import logging

router = APIRouter()

@router.post("/predictViralDisease")
async def predict_sequence_endpoint(sequence_input: SequenceInput):
    try:
        sequence = sequence_input.sequence
        if not sequence:
            raise HTTPException(status_code=400, detail="No sequence provided.")
        
        prediction = predict_sequence(sequence)
        return {"disease": prediction}
    
    except Exception as e:
        logging.error(f"Error predicting sequence: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.post("/predictFileViralDisease")
async def predict_file_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        predictions = predict_file(content.decode("utf-8"))
        return {"predictions": predictions}
    
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
