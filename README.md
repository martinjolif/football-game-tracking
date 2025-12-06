# football-game-tracking

The objective of this project is to generate game commentaries with LLMs from insights inferred from a video/livestream of a football game.

To do so, I need to detect and track the ball and the players on images obtained from the video. I also detect some pitch keypoints, and by homography can compute the position of the players and the ball on the pitch.

If you want to reproduce the training process for detecting players, ball, pitch keypoints, follow these [instructions](#local-installation-instructions).
If you only want to use the complete application, you can simply [pull the Docker image and run the container](#run-the-application-via-docker) to experiment with the full system (still in progress). 

## Local Installation Instructions
Clone the repository:

```
git clone git@github.com:martinjolif/football-game-tracking.git
cd football-game-tracking
```
If ``uv`` is not installed on your computer run:
```
curl -Ls https://astral.sh/uv/install.sh | sh
```

Create the `uv` environment using the following command:
```
uv venv
uv sync
```

### Access MLflow to track experiments
To get access to mlflow project, run the following command:
```
mlflow server \
  --backend-store-uri runs/mlflow \
  --host 127.0.0.1 \
  --port 5001
```
then open http://127.0.0.1:5001/. You will see here the model you trained with their corresponding parameters and metrics.

### Train & Evaluate ML models

#### Football player detection

From the ``football-game-tracking`` folder, run the following command:
```
PYTHONPATH=$PYTHONPATH:./src python training/player_detection/train.py
```
The ``train.py`` file will both train and evaluate the model on the test set. The `evaluation.py` file will choose a pretrained model and evaluate it only on the test set.

```
PYTHONPATH=$PYTHONPATH:./src python training/player_detection/evaluation.py
```
Look at the ``training/player_detection/config.py`` file to modify some training/evaluation parameters.


#### Football ball detection

From the ``football-game-tracking`` folder, run the following command:

```
PYTHONPATH=$PYTHONPATH:./src python training/ball_detection/train.py
```

```
PYTHONPATH=$PYTHONPATH:./src python training/ball_detection/evaluation.py
```
Look at the ``training/ball_detection/config.py`` file to modify some training/evaluation parameters.


#### Football pitch detection

From the ``football-game-tracking`` folder, run the following command to train the model:

```
PYTHONPATH=$PYTHONPATH:./src python training/pitch_detection/train.py
```

```
PYTHONPATH=$PYTHONPATH:./src python training/pitch_detection/evaluation.py
```
Look at the ``training/pitch_detection/config.py`` file to modify some training/evaluation parameters.

#### Team clustering
The goal is first to create a dataset of player crops labeled by their color jersey in order to train a classification model that will be able to classify jersey colors. Hopefully, from there some layers from the classification model can be reused to create embeddings for clustering players by their jersey color.

###### 1. Dataset creation
Extract frames from videos (SoccerNet dataset):
```
python src/team_clustering/extract_frames.py
```

Create player crops from extracted frames:
```
 PYTHONPATH=$PYTHONPATH:./src python training/team_clustering/crop_players.py
```

###### 2. Color jersey classification model training


###### 3. Team clustering with embeddings evaluation



### 2D Pitch Radar

(Still in progress)

The idea is from the players, ball and pitch keypoints detection to create a 2D pitch radar that show us the position of the players and the ball in the pitch. 

To do so, I need to track players and assign them an id, give them a class that corresponds to their team.

#### Player tracking

Set the variable ``PLAYER_TRACKING_VIZ`` to ``True`` in order to visualize the tracking output by running the ``video_to_frames.py`` script.

#### Team classification

Goal: identify player team from their corresponding crop images found by the player detection part. 

Unfortunately, identification via the average color pixels of the crop isn't working well due to several things: background (grass, stands, other players...), size of the crops vary a lot, lightning.


### Run tests
```
uv run pytest
```

### API Server Startup Instructions (Player, Ball, Pitch)
To run the API server, use the following command from the root folder:
```
uvicorn src.player_detection.api.app:app --reload --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/player-detection/image" -F "file=@<path_to_your_image.jpg>"
```

```
uvicorn src.ball_detection.api.app:app --reload --host 0.0.0.0 --port 8001
curl -X POST "http://localhost:8001/ball-detection/image" -F "file=@<path_to_your_image.jpg>"
```

```
uvicorn src.pitch_detection.api.app:app --reload --host 0.0.0.0 --port 8002
curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@<path_to_your_image.jpg>"
```
Make sure to update the ``src/player_detection/api/config.py`` file with the correct path to your model weights.

Call examples
```
curl -X POST "http://localhost:8000/player-detection/image" -F "file=@training/player_detection/data/yolov8-format/test/images/4b770a_9_3_png.rf.64599238d2f363e9e36e711b55426d1b.jpg"
curl -X POST "http://localhost:8001/ball-detection/image" -F "file=@training/ball_detection/data/yolov8-format/test/images/0a2d9b_0_mp4-0071_jpg.rf.852b629138f67394f68b712f3160b7a2.jpg"
curl -X POST "http://localhost:8002/pitch-detection/image" -F "file=@training/pitch_detection/data/yolov8-format/test/images/08fd33_2_9_png.rf.904829f5d75dafc562926ef44d02c5a3.jpg"
```

### Build and Run Docker Image
Build a multi-arch Docker image (ARM64 + AMD64):
```
docker buildx build --platform linux/arm64/v8,linux/amd64 \ 
    -t <docker-username>/football-game-tracking \
    -f docker/Dockerfile .
```
Run the container:
```
docker run -p 8000:8000 <docker-username>/football-game-tracking 
```
If your image exposes multiple services (ports 8000/8001/8002), you can expose them all:
```
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 \ 
    <docker-username>/football-game-tracking
```
Call the API:
```
curl -X POST "http://localhost:8000/player-detection/image" \
    -F "file=@<path_to_your_image.jpg>"
```

## Data 
If you want to test the full pipeline with a real football game video as input, you can download one of the ``Broadcast Videos`` from the ``SoccerNet`` dataset. You can download the videos from the following link: https://www.soccer-net.org/data by filling the NDA form.

## Run the Application via Docker

(The final app is still not available)

Pull the image:
```
docker pull mjolif/football-game-tracking
```
Run the container:
```
docker run -p 8000:8000 mjolif/football-game-tracking 
```
Call the API:
```
curl -X POST "http://<ip-address>:8000/player-detection/image" \
    -F "file=@<path_to_your_image.jpg>"
```