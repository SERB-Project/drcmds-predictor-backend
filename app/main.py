from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
import uvicorn
from app.api import router as api_router
from fastapi.staticfiles import StaticFiles
import os
import time
import threading
import schedule

app = FastAPI(
    title=settings.API_NAME,
    version=settings.API_VERSION
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def home():
    return {"health_check": "OK"}

# Define the cleanup function for temporary graph files
def cleanup_temp_files(max_age_hours=24):
    """Remove temp graph files older than the specified age"""
    temp_dir = os.path.join("app", "static", "temp")
    if not os.path.exists(temp_dir):
        return
    
    current_time = time.time()
    for filename in os.listdir(temp_dir):
        if filename.startswith("graph_") and filename.endswith(".png"):
            file_path = os.path.join(temp_dir, filename)
            # Get file's last modification time
            file_age = current_time - os.path.getmtime(file_path)
            # Convert seconds to hours
            if file_age > (max_age_hours * 3600):
                try:
                    os.remove(file_path)
                    print(f"Removed old file: {filename}")
                except Exception as e:
                    print(f"Failed to remove {filename}: {e}")

# Register startup event to initialize the cleanup scheduler
@app.on_event("startup")
async def startup_event():
    # Make sure the temp directory exists
    os.makedirs(os.path.join("app", "static", "temp"), exist_ok=True)
    
    # Run cleanup once a day
    schedule.every(24).hours.do(cleanup_temp_files)
    
    # Start the scheduler in a background thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    
    cleanup_thread = threading.Thread(target=run_scheduler)
    cleanup_thread.daemon = True  # Thread will exit when main program exits
    cleanup_thread.start()
    
    # Run cleanup immediately on startup
    cleanup_temp_files()
