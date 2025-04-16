from pydantic import BaseModel, Field
from typing import Optional

class SequenceInput(BaseModel):
    sequence: str = Field(..., description="Single DNA sequence input.")
