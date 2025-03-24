from pydantic import BaseModel

# Model for handling JSON input
class SequenceInput(BaseModel):
    sequence: str