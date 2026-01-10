import requests

from src.commentary_generation.config import LLM_MODEL, TEMPERATURE, TOP_P
from src.commentary_generation.events3 import generate_event
from src.utils.logger import LOGGER

def generate_prompt(previous_ball_xy, ball_xy, players_xy, cluster_labels, left_team, right_team, teams_barycenter, pitch):
    prompt = "You are a football commentator. Describe the following situation in 1-2 sentences:\n\n"
    event = generate_event(previous_ball_xy, ball_xy, players_xy, cluster_labels, left_team, right_team, teams_barycenter, pitch)
    if event is not None:
        prompt += event
        return prompt
    else:
        return None

def generate_commentary_ollama(previous_ball_xy, ball_xy, players_xy, cluster_labels, left_team, right_team, teams_barycenter, pitch, model=LLM_MODEL, temperature=TEMPERATURE, top_p=TOP_P):
    url = "http://localhost:11434/api/generate"
    prompt = generate_prompt(previous_ball_xy, ball_xy, players_xy, cluster_labels, left_team, right_team, teams_barycenter, pitch)
    if prompt is None:
        return None
    else:
        LOGGER.debug("Generating commentary with OLLAMA LLM model: %s", model)
        LOGGER.debug("------Prompt to the LLM------\n%s", prompt)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
