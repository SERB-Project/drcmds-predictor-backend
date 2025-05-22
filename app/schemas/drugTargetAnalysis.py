from pydantic import BaseModel
from typing import Optional, List

class DrugTargetInput(BaseModel):
    compound_smiles: str
    target_sequence: str

class PredictionResult(BaseModel):
    affinity: float
    explanation_graph: Optional[str] = None
