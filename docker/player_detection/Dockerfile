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

COPY pyproject.toml uv.lock .python-version ./

# Synchronize uv env
RUN uv venv && uv sync --no-dev

# Copy content to /app
COPY src/utils ./src/utils
COPY src/player_detection ./src/player_detection

RUN uv run hf download martinjolif/yolo-football-player-detection yolo-football-player-detection.pt \
    --local-dir weights/player_detection/hf_weights

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn",  "src.player_detection.api.app:app", "--host", "0.0.0.0", "--port",  "8000"]