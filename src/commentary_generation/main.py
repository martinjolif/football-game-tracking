import requests

from commentary_generation.config import LLM_MODEL
from src.commentary_generation.events import generate_event

def generate_prompt(ball_xy, players_xy, cluster_labels, pitch):
    prompt = "You are a football commentator. Describe the following situation in 1-2 sentences:\n\n"
    prompt += generate_event(ball_xy, players_xy, cluster_labels, pitch)
    return prompt

def generate_commentary_ollama(ball_xy, players_xy, cluster_labels, pitch, model=LLM_MODEL):
    url = "http://localhost:11434/api/generate"
    prompt = generate_prompt(ball_xy, players_xy, cluster_labels, pitch)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["response"]
