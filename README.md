# football-game-tracking

### Installation Instructions
clone the repository

```
git clone git@github.com:martinjolif/football-game-tracking.git
cd football-game-tracking
```

Create the `uv` environment using the following command:

```
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
```

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
PYTHONPATH=$PYTHONPATH:./src python src/player_detection/training/train.py
```
#### Evaluate model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python src/player_detection/training/evaluation.py
```

#### Running the API server
To run the API server, use the following command from the ``src`` folder:
```
uvicorn player_detection.api.app:app --reload --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/player-detection/image" -F "file=@path_to_your_image.jpg"
```
Example
```
 curl -X POST "http://localhost:8000/player-detection/image" -F "file=@player_detection/data/yolov8-format/test/images/4b770a_9_3_png.rf.64599238d2f363e9e36e711b55426d1b.jpg"`
```
### Football ball detection

#### Train model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python src/ball_detection/training/train.py
```
#### Evaluate model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python src/ball_detection/training/evaluation.py
```

#### Running the API server

To run the API server, use the following command from the ``src`` folder:
```
uvicorn ball_detection.api.app:app --reload --host 0.0.0.0 --port 8001
curl -X POST "http://localhost:8001/ball-detection/image" -F "file=@path_to_your_image.jpg"
```
Example:
```
curl -X POST "http://localhost:8001/ball-detection/image" -F "file=@ball_detection/data/yolov8-format/test/images/0a2d9b_0_mp4-0071_jpg.rf.852b629138f67394f68b712f3160b7a2.jpg"
```
### Football pitch detection

#### Train model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python src/pitch_detection/training/train.py
```
#### Evaluate model
From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python src/pitch_detection/training/evaluation.py
```
#### Running the API server

To run the API server, use the following command from the ``src`` folder:
```
uvicorn pitch_detection.api.app:app --reload --host 0.0.0.0 --port 8002
curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@path_to_your_image.jpg"
```
Example:
```
 curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@pitch_detection/data/yolov8-format/test/images/08fd33_2_9_png.rf.904829f5d75dafc562926ef44d02c5a3.jpg"
```