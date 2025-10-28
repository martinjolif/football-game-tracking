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

### Running the training script
To get access to mlflow project, run the following command:
```
 mlflow server --backend-store-uri football-game-tracking/runs/mlflow
```
then open http://127.0.0.1:5000/


