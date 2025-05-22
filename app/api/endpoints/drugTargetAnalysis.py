from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import logging
import csv
import io
import uuid

from app.schemas.drugTargetAnalysis import DrugTargetInput, PredictionResult
from app.services.drugTargetAnalysis.predict import predict_affinity

router = APIRouter()

@router.post("/predictDrugTargetAffinity", response_model=PredictionResult)
async def predict_drug_target_affinity(input_data: DrugTargetInput):
    try:
        if not input_data.compound_smiles:
            raise HTTPException(status_code=400, detail="No compound SMILES provided")
        
        # Generate a unique filename for the graph
        graph_filename = f"graph_{uuid.uuid4()}.png"
        temp_graph_path = f"temp/{graph_filename}"
        graph_path = f"app/static/temp/{graph_filename}"
        
        # Get prediction
        prediction, graph_encoding = predict_affinity(
            input_data.compound_smiles, 
            input_data.target_sequence,
            generate_graph=True
        )
        
        return {
            "affinity": prediction,
            "explanation_graph": graph_encoding
        }
    
    except ValueError as ve:
        logging.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logging.error(f"Error predicting affinity: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/getExplanationGraph/{graph_name}")
async def get_explanation_graph(graph_name: str):
    # Clean up the graph name (remove any path information)
    clean_name = os.path.basename(graph_name)
    graph_path = os.path.join("app", "static", "temp", clean_name)
    
    if not os.path.exists(graph_path):
        # For debugging
        print(f"Graph file not found at {graph_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in temp directory: {os.listdir(os.path.join('app', 'static', 'temp')) if os.path.exists(os.path.join('app', 'static', 'temp')) else 'temp dir not found'}")
        
        raise HTTPException(status_code=404, detail="Graph not found")
    
    return FileResponse(graph_path)


@router.post("/predictBatchDrugTargetAffinity")
async def predict_batch_affinities(file: UploadFile = File(...)):
    try:
        content = await file.read()
        content_str = content.decode("utf-8")
        
        # Parse CSV content
        csv_reader = csv.reader(io.StringIO(content_str))
        results = []
        
        for row in csv_reader:
            if len(row) < 1:
                continue
                
            compound_smiles = row[0]
            target_sequence = row[1] if len(row) > 1 else None
            
            try:
                prediction, _ = predict_affinity(compound_smiles, target_sequence)
                results.append({
                    "compound_smiles": compound_smiles,
                    "target_sequence": target_sequence,
                    "affinity": prediction
                })
            except Exception as e:
                results.append({
                    "compound_smiles": compound_smiles,
                    "target_sequence": target_sequence,
                    "error": str(e)
                })
        
        return {"predictions": results}
    
    except Exception as e:
        logging.error(f"Error processing batch file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
