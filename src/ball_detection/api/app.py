from fastapi import FastAPI
from ball_detection.api.router import router as ball_detection_router

app = FastAPI(title="Football Ball Detection API")
app.include_router(ball_detection_router)