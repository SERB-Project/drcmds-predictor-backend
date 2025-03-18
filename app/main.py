from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
import uvicorn
from app.api import router as api_router

app = FastAPI(
    title=settings.API_NAME,
    version=settings.API_VERSION
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router,prefix="/api")

@app.get("/")
def home():
    return {"health_check": "OK"}