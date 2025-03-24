from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from typing import Optional, Union
from app.services.sarsVariantsClassificationMutation.preprocess import preprocess_sequence as preprocess
from app.services.sarsVariantsClassificationMutation.predict import classify_variant as predict
from app.services.sarsVariantsClassificationMutation.parseFastaFiles import parse_fasta
from app.schemas.sarsPredictionReq import SequenceInput
import logging

router = APIRouter()

@router.post("/predictSarsSequence")
async def predict_sequence(sequence_input: SequenceInput):
    # Get sequence from JSON
    sequence = sequence_input.sequence
    
    # Preprocess and predict
    encoded_sequence = preprocess(sequence)
    prediction = predict(encoded_sequence)

    return {"variant": prediction}

@router.post("/predictSarsFile")
async def predict_file(file: UploadFile):
    try:
        content = await file.read()
        sequence = parse_fasta(content.decode("utf-8"))
        
        # Preprocess and predict
        encoded_sequence = preprocess(sequence)
        prediction = predict(encoded_sequence)

        return {"variant": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing FASTA file: {str(e)}")
