from pydantic import BaseModel
from typing import Optional

class SequenceInput(BaseModel):
    sequence: Optional[str] = None
