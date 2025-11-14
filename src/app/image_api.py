import io
import mimetypes
import os
from typing import Any, Dict, List, Optional, Union

import requests
from fastapi import HTTPException

def call_image_api(
        endpoint: str,
        image_path: Optional[str],
        image_bytes: Optional[bytes],
        timeout: int = 30
) -> dict | str:
    if image_bytes is None and image_path is None:
        raise ValueError("Either image_bytes or image_path must be provided.")
    try:
        if image_path:
            # From a local file
            content_type, _ = mimetypes.guess_type(image_path)
            if content_type is None:
                content_type = "application/octet-stream"
            try:
                with open(image_path, "rb") as f:
                    file_content = f.read()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
            except PermissionError:
                raise HTTPException(status_code=403, detail=f"Permission denied: {image_path}")
            files = {"file": (os.path.basename(image_path), file_content, content_type)}
            response = requests.post(endpoint, files=files, timeout=timeout)
        else:
            # From bytes (simulate file upload)
            files = {"file": ("frame.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            response = requests.post(endpoint, files=files, timeout=timeout)

        if response.status_code == 200:
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                return response.text
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except requests.exceptions.Timeout as e:
        raise HTTPException(status_code=504, detail=f"upstream timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=502, detail=f"upstream connection error: {e}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"upstream request failed: {e}")


def call_image_apis(
    endpoints: Optional[List[str]] = None,
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    timeout: int = 15
) -> Dict[str, Union[dict, str, Dict[str, Any]]]:
    """
    Sends the same image to several endpoints and returns a dictionary mapping each endpoint URL
    to its response (JSON, text, or error dict).
    By default, the endpoints are:
      - http://localhost:8000/player-detection/image
      - http://localhost:8001/ball-detection/image
      - http://localhost:8002/pitch-detection/image
    """
    if image_bytes is None and image_path is None:
        raise ValueError("Either image_bytes or image_path must be provided.")

    if endpoints is None:
        endpoints = [
            "http://localhost:8000/player-detection/image",
            "http://localhost:8001/ball-detection/image",
            "http://localhost:8002/pitch-detection/image",
        ]

    results: Dict[str, Dict[str, Any]] = {}
    for endpoint in endpoints:
        try:
            results[endpoint] = call_image_api(endpoint, image_path, image_bytes, timeout=timeout)
        except HTTPException as e:
            results[endpoint] = {"error": str(e.detail), "status_code": e.status_code}
    return results