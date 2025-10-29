from fastapi import FastAPI
from router import router as player_detection_router

app = FastAPI(title="Football Player Detection API")
app.include_router(player_detection_router)