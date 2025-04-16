from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.spliceSitePrediction import SequenceInput
from app.services.spliceSitePrediction.acceptor_services import predict_acceptor_sequence, predict_acceptor_file
from app.services.spliceSitePrediction.donor_services import predict_donor_sequence, predict_donor_file
import logging

router = APIRouter()

# Donor site sequence prediction
@router.post("/donor/predictSequence")
async def donor_predict_sequence_endpoint(data: SequenceInput):
    try:
        if not data.sequence:
            raise HTTPException(status_code=400, detail="No sequence provided")
        
        prediction = predict_donor_sequence(data.sequence)
        return {"prediction": prediction}
    
    except Exception as e:
        logging.error(f"Error predicting donor sequence: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Donor site file prediction
@router.post("/donor/predictFile")
async def donor_predict_file_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        predictions = predict_donor_file(content.decode("utf-8"))
        return {"predictions": predictions}
    
    except Exception as e:
        logging.error(f"Error processing donor file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


# Acceptor site sequence prediction
@router.post("/acceptor/predictSequence")
async def acceptor_predict_sequence_endpoint(data: SequenceInput):
    try:
        if not data.sequence:
            raise HTTPException(status_code=400, detail="No sequence provided")
        
        prediction = predict_acceptor_sequence(data.sequence)
        return {"prediction": prediction}
    
    except Exception as e:
        logging.error(f"Error predicting acceptor sequence: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Acceptor site file prediction
@router.post("/acceptor/predictFile")
async def acceptor_predict_file_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        predictions = predict_acceptor_file(content.decode("utf-8"))
        return {"predictions": predictions}
    
    except Exception as e:
        logging.error(f"Error processing acceptor file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
