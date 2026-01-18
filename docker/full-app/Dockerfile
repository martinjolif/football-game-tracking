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
COPY src/app ./src/app
COPY src/commentary_generation ./src/commentary_generation
COPY src/final_app ./src/final_app
COPY src/radar ./src/radar
COPY src/team_clustering ./src/team_clustering
COPY src/utils ./src/utils

# Create necessary directories
RUN mkdir -p uploads outputs

RUN uv run hf download martinjolif/mobilenetv3-football-jersey-classification \
    mobilenetv3-football-jersey-classification.pth --local-dir weights/team_clustering/hf_weights

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "src.final_app.main:app", "--host", "0.0.0.0", "--port", "8080"]