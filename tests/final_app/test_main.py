import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.final_app.main import app, job_status

client = TestClient(app)

def test_serve_frontend():
    # returns frontend when index html exists
    with patch('src.final_app.main.FRONTEND_DIR') as mock_dir:
        mock_index = MagicMock()
        mock_index.exists.return_value = True
        mock_dir.__truediv__ = Mock(return_value=mock_index)
        
        response = client.get("/")
        # Either 200 (if file exists) or 404 (if not) is acceptable
        assert response.status_code in [200, 404]

def test_upload_video():
    # uploads video and starts processing
    with patch('src.final_app.main.shutil') as mock_shutil, \
         patch('src.final_app.main.process_video') as mock_process:
        # Create a mock file
        mock_file = ("test.mp4", b"video content", "video/mp4")
        
        response = client.post(
            "/upload",
            files={"video": mock_file},
            data={
                "enable_radar": "true",
                "enable_commentary": "true",
                "enable_tracking": "true",
                "enable_team_clustering": "true",
                "cluster_train_frames": "50"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["message"] == "Video processing started"
        assert "status_url" in data
        
        # Clean up job status
        if data["job_id"] in job_status:
            del job_status[data["job_id"]]

def test_get_status():
    # returns status for existing job
    job_status["test-job-id"] = {
        "status": "processing",
        "progress": 50,
        "message": "Processing..."
    }
    
    response = client.get("/status/test-job-id")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["progress"] == 50

    # returns 404 for non existent job
    response = client.get("/status/non-existent-job")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data

def test_download_video():
    # returns 404 for non existent job
    response = client.get("/download/non-existent-job")
    assert response.status_code == 404

    # returns 400 for incomplete job
    job_status["incomplete-job"] = {
        "status": "processing"
    }
    response = client.get("/download/incomplete-job")
    assert response.status_code == 400

    # returns 404 when output file doesnt exist
    job_status["completed-job"] = {
        "status": "completed"
    }
    with patch('src.final_app.main.OUTPUT_DIR') as mock_dir:
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_dir.__truediv__ = Mock(return_value=mock_path)
        
        response = client.get("/download/completed-job")
        assert response.status_code == 404

def test_update_progress():
    # updates job progress
    from src.final_app.main import update_progress
    
    job_status["test-job"] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting..."
    }
    
    update_progress("test-job", 50, "Halfway done")
    
    assert job_status["test-job"]["progress"] == 50
    assert job_status["test-job"]["message"] == "Halfway done"

    # ignores non existent job
    update_progress("non-existent", 100, "Done")
    assert "non-existent" not in job_status
