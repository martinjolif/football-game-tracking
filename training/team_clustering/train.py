import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm

from training.logger import LOGGER
from training.team_clustering.utils import build_items_from_csv, JerseyDataset
from training.team_clustering.evaluation import evaluate_model

def train_one_epoch(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)

    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)

        batch_acc = (preds == targets).float().mean().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "batch_acc": f"{batch_acc:.4f}"})

    return running_loss / total, correct / total if total else 0.0

def evaluate(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> tuple[float, float]:
    """Evaluate the model on the validation set."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    pbar = tqdm(loader, desc="Val  ", leave=False)

    with torch.no_grad():
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)

            batch_acc = (preds == targets).float().mean().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "batch_acc": f"{batch_acc:.4f}"})

    return running_loss / total, correct / total if total else 0.0

parser = argparse.ArgumentParser(description="Train a model on player crops and their color labels.")
parser.add_argument(
    "--train-data-dir",
    default="training/team_clustering/data/crops-train",
    help="Root directory containing crop images (searched recursively)."
)
parser.add_argument(
    "--train-csv", default="crops_index.csv", help="CSV with columns `crop_image` and `color`"
)
parser.add_argument(
    "--test-data-dir",
    default="training/team_clustering/data/crops-test",
    help="Root directory containing crop images (searched recursively)."
)
parser.add_argument(
    "--test-csv", default="crops_index.csv", help="CSV with columns `crop_image` and `color`"
)
parser.add_argument(
    "--out-dir",
    default="training/team_clustering/models",
    help="Directory where the trained model and label mappings will be saved."
)
parser.add_argument("--batch-size", type=int, default=32, help="Number of samples per training batch.")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
parser.add_argument(
    "--val-split", type=float, default=0.2, help="Fraction of data to use for validation (0.0 - 1.0)."
)
parser.add_argument(
    "--img-size", type=int, default=224, help="Height and width to which images will be resized."
)

args = parser.parse_args()

data_dir = Path(args.train_data_dir)
csv_path = data_dir / args.train_csv
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

items = build_items_from_csv(csv_path, data_dir, crop_col="crop_image", label_col="color")
if not items:
    raise SystemExit("No labeled items found. Check CSV and data directory.")

labels = sorted({lbl for _, lbl in items})
label2idx = {lbl: i for i, lbl in enumerate(labels)}

train_items, val_items = train_test_split(
    items,
    test_size=args.val_split,
    stratify=[lbl for _, lbl in items] if len(labels) > 1 else None,
    random_state=42
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize(int(args.img_size * 256 / 224)),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
])

train_ds = JerseyDataset(train_items, label2idx, transform=train_transform)
val_ds = JerseyDataset(val_items, label2idx, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Preprocessing using MobileNetV3 weights statistics
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(labels))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_acc = 0.0
best_path = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
best_path.mkdir(parents=True, exist_ok=True)
best_model_path = best_path / "best_mobilenetv3_small.pth"
mapping_path = best_path / "label2idx.json"

mlflow.set_tracking_uri("file:runs/mlflow")
mlflow.set_experiment("training/team_clustering/models")

with mlflow.start_run():
    # Log high-level parameters
    mlflow.log_params({
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "val_split": args.val_split,
        "img_size": args.img_size,
        "num_labels": len(labels),
        "labels": labels,
        "model": "mobilenet_v3_small",
        "data_dir": str(args.train_data_dir),
    })

    # Save mapping
    with open(mapping_path, "w") as f:
        json.dump(label2idx, f, indent=2)
    mlflow.log_artifact(str(mapping_path))

    pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="ep")
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "label2idx": label2idx}, best_model_path)
            mlflow.log_artifact(str(best_model_path))

        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}"
        })

    mlflow.log_metric("best_val_acc", best_acc)
    LOGGER.info(f"Training finished. Best val_acc: {best_acc:.4f}")

    evaluate_model(
        model_path=best_model_path,
        csv_path=Path(args.test_data_dir) / args.test_csv,
        test_dir=args.test_data_dir,
        out_dir=best_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=device
    )

    # Log UMAP plot generated by evaluate_model
    umap_path = best_path / "umap_ground_truth.png"
    mlflow.log_artifact(str(umap_path))