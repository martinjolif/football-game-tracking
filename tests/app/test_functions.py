import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.functions import process_image

def test_process_image():
    # raises error when multiple flags are true
    with pytest.raises(ValueError, match="Only one debug or visualization flag can be True at a time."):
        process_image("test.jpg", player_tracking_viz=True, pitch_radar_viz=True)

    # raises error when image file not found
    with pytest.raises(FileNotFoundError, match="Cannot load image"):
        process_image("nonexistent.jpg")

    # processes image with debug player detection
    with patch('app.functions.cv2.imread') as mock_imread, \
         patch('app.functions.cv2.imencode') as mock_imencode, \
         patch('app.functions.call_image_apis') as mock_apis, \
         patch('app.functions.render_detection_results') as mock_render:
        
        # Setup mocks
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_frame
        mock_imencode.return_value = (True, np.array([1, 2, 3]))
        
        mock_apis.return_value = {
            "http://localhost:8000/player-detection/image": {
                "detections": [],
                "mapping_class": {"1": "player"}
            },
            "http://localhost:8001/ball-detection/image": {
                "detections": [],
                "mapping_class": {"1": "ball"}
            },
            "http://localhost:8002/pitch-detection/image": {
                "poses": []
            }
        }
        
        mock_render.return_value = mock_frame
        
        result = process_image("test.jpg", debug_player_detection=True)
        
        assert result is not None
        mock_render.assert_called_once()

    # processes image with debug all
    with patch('app.functions.cv2.imread') as mock_imread, \
         patch('app.functions.cv2.imencode') as mock_imencode, \
         patch('app.functions.call_image_apis') as mock_apis, \
         patch('app.functions.render_detection_results') as mock_render:
        
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_frame
        mock_imencode.return_value = (True, np.array([1, 2, 3]))
        
        mock_apis.return_value = {
            "http://localhost:8000/player-detection/image": {
                "detections": [],
                "mapping_class": {"1": "player"}
            },
            "http://localhost:8001/ball-detection/image": {
                "detections": [],
                "mapping_class": {"1": "ball"}
            },
            "http://localhost:8002/pitch-detection/image": {
                "poses": []
            }
        }
        
        mock_render.return_value = mock_frame
        
        result = process_image("test.jpg", debug_all=True, debug_player_detection=False)
        
        assert result is not None
        mock_render.assert_called_once()

    # processes image with player tracking viz
    with patch('app.functions.cv2.imread') as mock_imread, \
         patch('app.functions.cv2.imencode') as mock_imencode, \
         patch('app.functions.call_image_apis') as mock_apis, \
         patch('app.functions.visualize_frame') as mock_viz, \
         patch('app.functions.sv.ByteTrack') as mock_tracker:
        
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_frame
        mock_imencode.return_value = (True, np.array([1, 2, 3]))
        
        mock_apis.return_value = {
            "http://localhost:8000/player-detection/image": {
                "detections": [],
                "mapping_class": {"1": "player"}
            },
            "http://localhost:8001/ball-detection/image": {
                "detections": [],
                "mapping_class": {"1": "ball"}
            },
            "http://localhost:8002/pitch-detection/image": {
                "poses": []
            }
        }
        
        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance
        mock_viz.return_value = mock_frame
        
        result = process_image("test.jpg", player_tracking_viz=True, debug_player_detection=False)
        
        assert result is not None
        mock_viz.assert_called_once()
        mock_tracker_instance.update_with_detections.assert_called_once()

    # processes image with pitch radar viz
    with patch('app.functions.cv2.imread') as mock_imread, \
         patch('app.functions.cv2.imencode') as mock_imencode, \
         patch('app.functions.call_image_apis') as mock_apis, \
         patch('app.functions.render_pitch_radar') as mock_radar, \
         patch('app.functions.sv.resize_image') as mock_resize, \
         patch('app.functions.sv.draw_image') as mock_draw:
        
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_frame
        mock_imencode.return_value = (True, np.array([1, 2, 3]))
        
        mock_apis.return_value = {
            "http://localhost:8000/player-detection/image": {
                "detections": [],
                "mapping_class": {"1": "player"}
            },
            "http://localhost:8001/ball-detection/image": {
                "detections": [],
                "mapping_class": {"1": "ball"}
            },
            "http://localhost:8002/pitch-detection/image": {
                "poses": []
            }
        }
        
        mock_radar_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_radar.return_value = (mock_radar_frame, None, None)
        mock_resize.return_value = mock_radar_frame
        mock_draw.return_value = mock_frame
        
        result = process_image("test.jpg", pitch_radar_viz=True, debug_player_detection=False)
        
        assert result is not None
        mock_radar.assert_called_once()
