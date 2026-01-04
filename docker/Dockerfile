FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
LABEL authors="mjolif"

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Download and install UV tool, then cleanup installer
RUN curl -fsSL https://astral.sh/uv/install.sh -o /uv-installer.sh \
    && sh /uv-installer.sh \
    && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Commands will be run from /app
WORKDIR /app

# Copy content to /app
COPY . /app

# Synchronize uv env
RUN uv venv && uv sync --no-dev

RUN uv run hf download martinjolif/yolo-football-player-detection --local-dir weights/player_detection/hf_weights \
    && uv run hf download martinjolif/mobilenetv3-football-jersey-classification --local-dir weights/team_clustering/hf_weights

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn",  "src.player_detection.api.app:app", "--host", "0.0.0.0", "--port",  "8000"]