from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List
import os

# Load environment variables from .env
load_dotenv()

class Settings(BaseSettings):
    API_NAME: str = os.getenv("API_NAME", "DrCMD_Predictor")
    API_VERSION: str = os.getenv("API_VERSION", "1.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # CORS settings
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", '["http://localhost:3000"]').strip("[]").replace('"', '').split(",")

# Create an instance of settings
settings = Settings()
