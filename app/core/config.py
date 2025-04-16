from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List
import os
import json

# Load environment variables from .env
load_dotenv()

class Settings(BaseSettings):
    API_NAME: str = os.getenv("API_NAME", "DrCMD_Predictor")
    API_VERSION: str = os.getenv("API_VERSION", "1.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # CORS settings
    CORS_ORIGINS: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cors_origins_env = os.getenv("CORS_ORIGINS", '["http://localhost:3000"]')
        try:
            self.CORS_ORIGINS = json.loads(cors_origins_env)
        except json.JSONDecodeError:
            self.CORS_ORIGINS = cors_origins_env.split(",")

# Create an instance of settings
settings = Settings()
