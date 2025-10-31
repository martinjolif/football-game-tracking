# football-game-tracking

### Football pitch detection

#### Train model
```
PYTHONPATH=$PYTHONPATH:./src python src/pitch_detection/training/train.py
```
#### Evaluate model
```
PYTHONPATH=$PYTHONPATH:./src python src/pitch_detection/training/evaluation.py
```

#### Running the API server

To run the API server, use the following command from the ``src`` folder:
```
uvicorn app:app --reload --host 0.0.0.0 --port 8002
curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@path_to_your_image.jpg"
```