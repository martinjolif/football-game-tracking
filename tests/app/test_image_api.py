import pytest
from unittest.mock import Mock, patch, mock_open
from fastapi import HTTPException
from app.image_api import call_image_api, call_image_apis

def test_call_image_api():
    # raises error when both image_bytes and image_path are none
    with pytest.raises(ValueError, match="Either image_bytes or image_path must be provided."):
        call_image_api("http://localhost:8000/test", None, None)

    # calls api with image bytes
    with patch('app.image_api.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response
        
        result = call_image_api("http://localhost:8000/test", None, b"image_data")
        assert result == {"result": "success"}
        mock_post.assert_called_once()

    # calls api with image path
    with patch('app.image_api.requests.post') as mock_post, \
         patch('builtins.open', mock_open(read_data=b"file_content")):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response
        
        result = call_image_api("http://localhost:8000/test", "/path/to/image.jpg", None)
        assert result == {"result": "success"}
        mock_post.assert_called_once()

    # raises http exception on non 200 status
    with patch('app.image_api.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_post.return_value = mock_response
        
        with pytest.raises(HTTPException) as exc_info:
            call_image_api("http://localhost:8000/test", None, b"image_data")
        assert exc_info.value.status_code == 404

    # returns text when json decode fails
    with patch('app.image_api.requests.post') as mock_post:
        import requests
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("JSON decode error", "", 0)
        mock_response.text = "plain text response"
        mock_post.return_value = mock_response
        
        result = call_image_api("http://localhost:8000/test", None, b"image_data")
        assert result == "plain text response"

    # handles timeout error
    with patch('app.image_api.requests.post') as mock_post:
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")
        
        with pytest.raises(HTTPException) as exc_info:
            call_image_api("http://localhost:8000/test", None, b"image_data")
        assert exc_info.value.status_code == 504

    # handles connection error
    with patch('app.image_api.requests.post') as mock_post:
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(HTTPException) as exc_info:
            call_image_api("http://localhost:8000/test", None, b"image_data")
        assert exc_info.value.status_code == 502


def test_call_image_apis():
    # raises error when both image_bytes and image_path are none
    with pytest.raises(ValueError, match="Either image_bytes or image_path must be provided."):
        call_image_apis(None, None, None)

    # calls multiple endpoints with default endpoints
    with patch('app.image_api.call_image_api') as mock_call:
        mock_call.return_value = {"result": "success"}
        
        results = call_image_apis(None, None, b"image_data")
        assert len(results) == 3
        assert "http://localhost:8000/player-detection/image" in results
        assert "http://localhost:8001/ball-detection/image" in results
        assert "http://localhost:8002/pitch-detection/image" in results

    # calls custom endpoints
    with patch('app.image_api.call_image_api') as mock_call:
        mock_call.return_value = {"result": "success"}
        
        endpoints = ["http://localhost:9000/test1", "http://localhost:9001/test2"]
        results = call_image_apis(endpoints, None, b"image_data")
        assert len(results) == 2
        assert "http://localhost:9000/test1" in results
        assert "http://localhost:9001/test2" in results

    # handles errors from individual endpoints
    with patch('app.image_api.call_image_api') as mock_call:
        def side_effect(endpoint, path, bytes, timeout):
            if "8000" in endpoint:
                return {"result": "success"}
            else:
                raise HTTPException(status_code=500, detail="Server error")
        
        mock_call.side_effect = side_effect
        
        results = call_image_apis(None, None, b"image_data")
        assert results["http://localhost:8000/player-detection/image"] == {"result": "success"}
        assert "error" in results["http://localhost:8001/ball-detection/image"]
        assert results["http://localhost:8001/ball-detection/image"]["status_code"] == 500
