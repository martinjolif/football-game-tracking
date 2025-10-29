# football-game-tracking


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
