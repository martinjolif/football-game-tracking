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
COPY src/pitch_detection ./src/pitch_detection

RUN uv run hf download martinjolif/yolo-football-pitch-detection yolo-football-pitch-detection.pt \
    --local-dir weights/pitch_detection/hf_weights

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn",  "src.pitch_detection.api.app:app", "--host", "0.0.0.0", "--port",  "8002"]