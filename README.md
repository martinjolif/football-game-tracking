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
### Football player detection
#### Running the training script
To get access to mlflow project, run the following command:
```
 mlflow server --backend-store-uri football-game-tracking/runs/mlflow
```
then open http://127.0.0.1:5000/


#### Running the API server
To run the API server, use the following command from the ``player_detection/api`` folder:
```
uvicorn app:app --reload --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/player-detection/image" -F "file=@path_to_your_image.jpg"
```
### Football ball detection

#### Train model
```
PYTHONPATH=$PYTHONPATH:./src python src/ball_detection/training/train.py
```
#### Evaluate model
```
PYTHONPATH=$PYTHONPATH:./src python src/ball_detection/training/evaluation.py
```

#### Running the API server

To run the API server, use the following command from the ``src`` folder:
```
uvicorn app:app --reload --host 0.0.0.0 --port 8001
curl -X POST "http://localhost:8001/ball-detection/image" -F "file=@path_to_your_image.jpg"
```
### Football pitch detection

#### Train model
```
PYTHONPATH=$PYTHONPATH:./src python src/pitch_detection/training/train.py
```
#### Evaluate model
```
PYTHONPATH=$PYTHONPATH:./src python src/pitch_detection/training/evaluation.py

#### Running the API server

To run the API server, use the following command from the ``src`` folder:
```
uvicorn pitch_detection.api.app:app --reload --host 0.0.0.0 --port 8002
curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@path_to_your_image.jpg"
```