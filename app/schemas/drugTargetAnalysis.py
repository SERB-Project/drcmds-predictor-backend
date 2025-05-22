from pydantic import BaseModel
from typing import Optional, List

class DrugTargetInput(BaseModel):
    compound_smiles: str
    target_sequence: Optional[str] = None

class PredictionResult(BaseModel):
    affinity: float
    explanation_graph: Optional[str] = None
