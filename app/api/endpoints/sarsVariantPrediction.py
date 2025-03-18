from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.sarsVariantsClassificationMutation.preprocess import preprocess_sequence as preprocess
from app.services.sarsVariantsClassificationMutation.predict import classify_variant as predict
import logging

router = APIRouter()

@router.post("/predictSarsClassificationMutations")
async def predict_variant(file: UploadFile = None, sequence: str = None):
    """
    Classify SARS-CoV-2 variant from sequence data.
    Accepts either a FASTA file or a raw sequence string.
    """
    try:
        if file:
            contents = await file.read()
            sequence = contents.decode().strip()
        
        if not sequence:
            raise HTTPException(status_code=400, detail="No sequence provided.")

        # Preprocess and predict
        encoded_sequence = preprocess(sequence)
        prediction = predict(encoded_sequence)

        return {"variant": prediction}
    
    except Exception as e:
        logging.error(f"🚨 Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
