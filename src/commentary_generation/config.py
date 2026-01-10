import os

LLM_MODEL = os.getenv("OLLAM_MODEL", "smollm2:1.7b")
TEMPERATURE = 0.7
TOP_P = 0.8