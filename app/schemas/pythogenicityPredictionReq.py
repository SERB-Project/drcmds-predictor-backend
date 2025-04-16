from pydantic import BaseModel
from typing import List

# Input schema
class PathogenicityInput(BaseModel):
    spdi: str
    consequences: List[str]