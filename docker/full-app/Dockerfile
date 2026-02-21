FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
LABEL authors="mjolif"

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg libsm6 libxext6 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Download and install UV tool, then cleanup installer
RUN curl -fsSL https://astral.sh/uv/install.sh -o /uv-installer.sh \
    && sh /uv-installer.sh \
    && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Commands will be run from /app
WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./

# Synchronize uv env
RUN uv venv && uv sync --no-dev

# Copy content to /app
COPY src/app ./src/app
COPY src/ball_detection ./src/ball_detection
COPY src/commentary_generation ./src/commentary_generation
COPY src/final_app ./src/final_app
COPY src/pitch_detection ./src/pitch_detection
COPY src/player_detection ./src/player_detection
COPY src/radar ./src/radar
COPY src/team_clustering ./src/team_clustering
COPY src/utils ./src/utils

# Create necessary directories
RUN mkdir -p uploads outputs

RUN uv run hf download martinjolif/yolo-football-player-detection \
    yolo-football-player-detection.pt --local-dir weights/player_detection/hf_weights

RUN uv run hf download martinjolif/yolo-football-ball-detection \
    yolo-football-ball-detection.pt --local-dir weights/ball_detection/hf_weights

RUN uv run hf download martinjolif/yolo-football-pitch-detection \
    yolo-football-pitch-detection.pt --local-dir weights/pitch_detection/hf_weights

RUN uv run hf download martinjolif/mobilenetv3-football-jersey-classification \
    mobilenetv3-football-jersey-classification.pth --local-dir weights/team_clustering/hf_weights

RUN uv run hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --local-dir weights/tts/hf_weights

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "src.final_app.main:app", "--host", "0.0.0.0", "--port", "8080"]