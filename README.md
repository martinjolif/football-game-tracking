# football-game-tracking

### Local Installation Instructions
clone the repository

```
git clone git@github.com:martinjolif/football-game-tracking.git
cd football-game-tracking
```

Create the `uv` environment using the following command:

```
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv
uv sync
```

### Docker Installation
clone the repository

```
git clone git@github.com:martinjolif/football-game-tracking.git
cd football-game-tracking
docker buildx build --platform linux/arm64/v8,linux/amd64 -t <docker-username>/football-game-tracking -f docker/Dockerfile .
docker run -p 8000:8000 <docker-username>/football-game-tracking   
```


### Data 
If you want to test the full pipeline with a real football game video as input, you need to download one of the ``Broadcast Videos`` from the ``SoccerNet`` dataset. You can download the videos from the following link: https://www.soccer-net.org/data by filling the NDA form.
After downloading the video, place it in the ``todo`` folder.

### Acces MLflow to track experiments
To get access to mlflow project, run the following command:
```
 mlflow server --backend-store-uri football-game-tracking/runs/mlflow
```
then open http://127.0.0.1:5000/

### Football player detection

#### Train model
From the ``football-game-tracking`` folder, run the following command:
```
PYTHONPATH=$PYTHONPATH:./src python training/player_detection/train.py
```
#### Evaluate model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python training/player_detection/evaluation.py
```

#### Running the API server
To run the API server, use the following command from the root folder:
```
uvicorn src.player_detection.api.app:app --reload --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/player-detection/image" -F "file=@path_to_your_image.jpg"
```
Example
```
 curl -X POST "http://localhost:8000/player-detection/image" -F "file=@training/player_detection/data/yolov8-format/test/images/4b770a_9_3_png.rf.64599238d2f363e9e36e711b55426d1b.jpg"
```
### Football ball detection

#### Train model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python training/ball_detection/train.py
```
#### Evaluate model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python training/ball_detection/evaluation.py
```

#### Running the API server

To run the API server, use the following command from the root folder:
```
uvicorn src.ball_detection.api.app:app --reload --host 0.0.0.0 --port 8001
curl -X POST "http://localhost:8001/ball-detection/image" -F "file=@path_to_your_image.jpg"
```
Example:
```
curl -X POST "http://localhost:8001/ball-detection/image" -F "file=@training/ball_detection/data/yolov8-format/test/images/0a2d9b_0_mp4-0071_jpg.rf.852b629138f67394f68b712f3160b7a2.jpg"
```
### Football pitch detection

#### Train model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python training/pitch_detection/train.py
```
#### Evaluate model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python training/pitch_detection/evaluation.py
```
#### Running the API server

To run the API server, use the following command from the root folder:
```
uvicorn src.pitch_detection.api.app:app --reload --host 0.0.0.0 --port 8002
curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@path_to_your_image.jpg"
```
Example:
```
 curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@training/pitch_detection/data/yolov8-format/test/images/08fd33_2_9_png.rf.904829f5d75dafc562926ef44d02c5a3.jpg"
```

### Player tracking

Set the variable ``PLAYER_TRACKING_VIZ`` to ``True`` in order to visualize the tracking output by running the ``video_to_frames.py`` script.

### Team classification

Goal: identify player team from their corresponding crop images found by the player detection part. 

Unfortunately, identification via the average color pixels of the crop isn't working well due to several things: background (grass, stands, other players...), size of the crops vary a lot, lightning.

### 2D pitch radar  