FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
LABEL authors="mjolif"

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg libsm6 libxext6

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Commands will be run from /app
WORKDIR /app

# Copy content to /app
COPY . /app

# Synchronize uv env
RUN uv venv && . .venv/bin/activate && uv sync --no-dev

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn",  "src.player_detection.api.app:app",  "--reload", "--host", "0.0.0.0", "--port",  "8000"]