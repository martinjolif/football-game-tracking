# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Football game tracking system that processes video to detect players, ball, and pitch, then generates live commentary with LLMs and TTS. The pipeline: YOLO detection → player tracking → team clustering → 2D pitch radar via homography → Ollama LLM commentary → Qwen3-TTS audio.

## Environment Setup

```bash
uv venv && uv sync
source .venv/bin/activate
```

All scripts require `PYTHONPATH=$PYTHONPATH:./src` to resolve `src.*` imports.

## Common Commands

**Run tests:**
```bash
uv run pytest
# Single test file:
PYTHONPATH=$PYTHONPATH:./src uv run pytest tests/app/test_functions.py
```

**Start detection API servers (run from repo root):**
```bash
uvicorn src.player_detection.api.app:app --reload --host 0.0.0.0 --port 8000
uvicorn src.ball_detection.api.app:app --reload --host 0.0.0.0 --port 8001
uvicorn src.pitch_detection.api.app:app --reload --host 0.0.0.0 --port 8002
```

**Start the final full app:**
```bash
uvicorn src.final_app.main:app --reload --host 0.0.0.0 --port 8080
```

**Train ML models:**
```bash
PYTHONPATH=$PYTHONPATH:./src python training/player_detection/train.py
PYTHONPATH=$PYTHONPATH:./src python training/ball_detection/train.py
PYTHONPATH=$PYTHONPATH:./src python training/pitch_detection/train.py
```

**Download model weights from HuggingFace:**
```bash
uv run hf download martinjolif/yolo-football-player-detection --local-dir weights/player_detection/hf_weights
uv run hf download martinjolif/yolo-football-ball-detection --local-dir weights/ball_detection/hf_weights
uv run hf download martinjolif/yolo-football-pitch-detection --local-dir weights/pitch_detection/hf_weights
uv run hf download martinjolif/mobilenetv3-football-jersey-classification --local-dir weights/team_clustering/hf_weights
```

**Docker (GPU):**
```bash
docker compose -f docker-compose.nvidia.yml build
docker compose -f docker-compose.nvidia.yml up -d
# App at http://localhost:8080
```

## Architecture

### Detection Microservices (`src/{player,ball,pitch}_detection/api/`)

Three independent FastAPI apps, each following the same pattern:
- `app.py` – FastAPI entrypoint
- `router.py` – endpoint definitions
- `inference.py` – YOLO model inference logic
- `config.py` – model path and thresholds (update these with correct weight paths)

All return `DetectionInferenceResponse` or `PoseInferenceResponse` (defined in `src/utils/schemas.py`).

### Final App (`src/final_app/`)

- `main.py` – FastAPI server that accepts video uploads, spawns background jobs, serves the frontend from `src/final_app/frontend/`
- `video_processor.py` – `VideoProcessor` class: the core processing loop. For each frame it:
  1. Calls the three detection APIs via `src/app/image_api.py`
  2. Updates ByteTrack player tracker
  3. Crops player bounding boxes and runs team clustering (MobileNetV3 + PCA + KMeans)
  4. Calls `render_pitch_radar()` to project detections onto a 2D pitch overlay using homography
  5. Generates commentary via Ollama (background thread) when ball position changes significantly
  6. Generates TTS audio via Qwen3-TTS (background thread)
  7. Assembles final video with audio via ffmpeg

Detection API URLs are configurable via env vars: `PLAYER_DETECTION_URL`, `BALL_DETECTION_URL`, `PITCH_DETECTION_URL`.

### Commentary Generation (`src/commentary_generation/`)

- `events3.py` – extracts game events from positions (ball zone, possession team, ball direction)
- `main.py` – builds prompt and calls Ollama (`OLLAMA_URL` env var, model from `OLLAM_MODEL` env var — note the typo in the env var name)
- `tts.py` – Qwen3-TTS singleton; env vars: `QWEN_TTS_MODEL`, `TTS_SPEAKER`, `TTS_LANGUAGE`
- `audio_assembler.py` – places TTS audio clips at the correct timestamps in the output video

### Radar / Homography (`src/radar/`)

- `homography.py` – `Homography` class wrapping `cv2.findHomography` + `cv2.perspectiveTransform`
- `pitch_dimensions.py` – standard pitch size constants
- `pitch_radar_visualization.py` – renders the 2D overhead pitch with player/ball positions

### Team Clustering (`src/team_clustering/`)

MobileNetV3 feature extractor → PCA(10) → KMeans(2). The model is trained on the first `cluster_train_frames` frames of each video. Per-tracker-id history smoothing corrects noisy per-frame predictions.

### Shared Utilities (`src/app/`)

- `image_api.py` – calls all detection endpoints in parallel
- `api_to_supervision.py` – converts API responses to `supervision` `Detections` objects
- `utils.py` – helper for extracting class IDs from API responses

## Key Configuration Notes

- Model weights must exist under `weights/` before running APIs or the final app (see download commands above)
- The team clustering model path defaults to `weights/team_clustering/hf_weights/mobilenetv3-football-jersey-classification.pth`
- TTS requires a CUDA GPU (`device_map="cuda:0"`); it is silently disabled if unavailable
- `ffmpeg` must be installed for the final video re-encoding step
- `pytest` configuration in `pyproject.toml` sets `pythonpath = ["src"]`, so tests can import `src.*` directly without the env var