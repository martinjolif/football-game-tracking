# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered football game analysis pipeline that processes video to detect players, track ball movement, identify teams by jersey color, compute pitch positions via homography, and generate natural language commentary using LLMs.

## Package Management

Uses **UV** (not pip). Always use `uv` commands:

```bash
uv venv && uv sync       # Setup environment
source .venv/bin/activate
uv run pytest            # Run tests
uv run <script.py>       # Run scripts
```

Python version: 3.13 (see `.python-version`)

## Common Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/app/test_functions.py

# Run a single test
uv run pytest tests/app/test_functions.py::test_name

# Start the full app locally (all detection endpoints included)
uvicorn src.final_app.main:app --reload --host 0.0.0.0 --port 8080

# Download model weights from HuggingFace
uv run hf download martinjolif/yolo-football-player-detection --local-dir weights/player_detection/hf_weights
uv run hf download martinjolif/yolo-football-ball-detection --local-dir weights/ball_detection/hf_weights
uv run hf download martinjolif/yolo-football-pitch-detection --local-dir weights/pitch_detection/hf_weights
uv run hf download martinjolif/mobilenetv3-football-jersey-classification --local-dir weights/team_clustering/hf_weights

# Docker (CPU)
docker compose -f docker-compose.cpu.yml build
docker compose -f docker-compose.cpu.yml up -d

# Docker (NVIDIA GPU)
docker compose -f docker-compose.nvidia.yml build
docker compose -f docker-compose.nvidia.yml up -d
```

## Architecture

Two services: the main app (all detection + orchestration) and OLLAMA (LLM inference).

### Services & Ports

| Service | Port | Purpose |
|---------|------|---------|
| Main app (`src/final_app/`) | 8080 | Video upload, job orchestration, detection endpoints, result download |
| OLLAMA | 11434 | LLM inference (smollm2:1.7b) |

Detection (player, ball, pitch) runs **in-process** via direct function calls — no HTTP round-trips per frame. The detection endpoints are still exposed on port 8080 (`/player-detection/image`, `/ball-detection/image`, `/pitch-detection/image`) for external use.

### Pipeline Flow (per-frame)

1. **Frame capture** → encode as JPEG
2. **Direct inference** → call player/ball/pitch detection functions in-process
3. **Tracking** → ByteTrack updates tracker IDs
4. **Team clustering** (frames 1–50: train UMAP+KMeans; frames 51+: infer)
5. **Homography** → project image coords to FIFA pitch coords (105m × 68m)
6. **Event detection** → possession, ball zone (3×3 grid), ball displacement
7. **Commentary** → LLM call if ball moved >200cm
8. **Rendering** → draw bboxes, radar overlay, commentary text
9. **Write frame** to output video

### Key Files

- `src/final_app/video_processor.py` — `VideoProcessor` class, main orchestration logic
- `src/final_app/main.py` — FastAPI server: upload endpoint, detection endpoints, job status polling, download
- `src/player_detection/api/inference.py` — Player detection (YOLO11m)
- `src/ball_detection/api/inference.py` — Ball detection (YOLO11)
- `src/pitch_detection/api/inference.py` — Pitch keypoint detection (YOLO11 pose)
- `src/app/api_to_supervision.py` — Converts inference results to `supervision` format
- `src/commentary_generation/events3.py` — Event extraction: team assignment, possession, zones
- `src/team_clustering/clustering_model.py` — MobileNetV3 + UMAP + KMeans pipeline
- `src/radar/homography.py` — 2D pitch coordinate transformation
- `src/utils/schemas.py` — Shared Pydantic models (`Detection`, `Pose`, `BoundingBox`)

### Detection API Response Schema

All three detection services return this shape:
```json
{
  "detections": [
    {"detected_class_id": 0, "confidence": 0.92, "bbox": {"x0": 10, "y0": 20, "x1": 50, "y1": 80}}
  ],
  "mapping_class": {"0": "player", "1": "goalkeeper", "2": "referee"},
  "inference_time": 0.045
}
```

### Environment Variables

For local development (set in `.env` or shell):
```
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=smollm2:1.7b
```

### Model Weights Paths

```
weights/player_detection/hf_weights/yolo-football-player-detection.pt
weights/ball_detection/hf_weights/yolo-football-ball-detection.pt
weights/pitch_detection/hf_weights/yolo-football-pitch-detection.pt
weights/team_clustering/hf_weights/mobilenetv3-football-jersey-classification.pth
```

### Test Configuration

pytest is configured in `pyproject.toml`:
- `testpaths = ["tests"]`
- `pythonpath = ["src"]` — imports resolve against `src/`

### CI/CD

GitHub Actions (`.github/workflows/ci-cd.yaml`):
1. `tests` job: runs `uv run pytest` on ubuntu-22.04
2. `build-docker-image` job: builds and pushes Docker images to Docker Hub (some builds commented out due to runner disk space constraints)
