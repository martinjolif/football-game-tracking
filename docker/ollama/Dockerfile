FROM ollama/ollama:latest

ARG OLLAMA_MODEL

# Pull the model during build
RUN ollama serve & \
    sleep 5 && \
    ollama pull ${OLLAMA_MODEL} && \
    pkill ollama

# Default command
CMD ["serve"]