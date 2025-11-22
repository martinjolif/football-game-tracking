from fastapi import FastAPI
from src.pitch_detection.api.router import router as pitch_detection_router

app = FastAPI(title="Football Pitch Detection API")
app.include_router(pitch_detection_router)