from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
import uvicorn

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

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.on_event("startup")
async def startup_event():
    print("\n🚀 Server started successfully!")
    print(f"🔗 Swagger Docs: http://localhost:{settings.PORT}/docs\n")
