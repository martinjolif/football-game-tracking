FROM ollama/ollama:0.13.5

ARG OLLAMA_MODEL="smollm:1.7b"

# Pull the model during build
RUN ollama serve & \
    sleep 5 && \
    ollama pull ${OLLAMA_MODEL} && \
    pkill ollama

# Default command
CMD ["serve"]