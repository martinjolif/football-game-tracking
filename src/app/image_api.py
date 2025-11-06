import mimetypes
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import HTTPException


def call_image_api(url: str, image_path: str, timeout: int = 30) -> dict | str:
    content_type, _ = mimetypes.guess_type(image_path)
    if content_type is None:
        content_type = 'application/octet-stream'
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"file not found: {image_path}")

    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, content_type)}
            try:
                response = requests.post(url, files=files, timeout=timeout)
            except requests.exceptions.Timeout as e:
                raise HTTPException(status_code=504, detail=f"upstream timeout: {e}")
            except requests.exceptions.ConnectionError as e:
                raise HTTPException(status_code=502, detail=f"upstream connection error: {e}")
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=502, detail=f"upstream request failed: {e}")

            if response.status_code == 200:
                try:
                    return response.json()
                except requests.exceptions.JSONDecodeError:
                    return response.text
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)

    except PermissionError as e:
        raise HTTPException(status_code=400, detail=f"permission denied reading file: {e}")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"cannot read file: {e}")


def call_image_apis(
    image_path: str,
    endpoints: Optional[List[str]] = None,
    timeout: int = 15
) -> Dict[str, Dict[str, Any]]:
    """
    Send the same image to several endpoints and return a list of responses.
    By default, the endpoints are:
      - http://localhost:8000/player-detection/image
      - http://localhost:8001/ball-detection/image
      - http://localhost:8002/pitch-detection/image
    """
    if endpoints is None:
        endpoints = [
            "http://localhost:8000/player-detection/image",
            "http://localhost:8001/ball-detection/image",
            "http://localhost:8002/pitch-detection/image",
        ]

    results: Dict[str, Dict[str, Any]] = {}
    for endpoint in endpoints:
        results[endpoint] = call_image_api(endpoint, image_path, timeout=timeout)
    return results