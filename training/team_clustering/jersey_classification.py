import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm

class JerseyDataset(Dataset):
    def __init__(self, items, label2idx, transform=None):
        """
        items: list of (image_path, label_str)
        label2idx: mapping label string -> int
        transform: torchvision transforms
        """
        self.items = items
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.label2idx[label]
        return img, target

def find_image(data_dir: Path, crop_name: str):
    """
    Try direct join, else search recursively by basename.
    Returns Path or None.
    """
    cand = data_dir / crop_name
    if cand.exists():
        return cand
    # fallback: search for basename
    basename = Path(crop_name).name
    for p in data_dir.rglob(basename):
        return p
    return None

def resolve_csv(csv_path: Path, data_dir: Path) -> Path | None:
    """
    Return a Path to the CSV if found, else None.
    Search order:
    1. Provided csv_path
    2. data_dir / csv_filename
    3. recursive search under data_dir
    4. recursive search under current working directory
    """
    if csv_path.exists():
        return csv_path
    alt = data_dir / csv_path.name
    if alt.exists():
        return alt
    for p in data_dir.rglob(csv_path.name):
        return p
    for p in Path.cwd().rglob(csv_path.name):
        return p
    return None

def build_items_from_csv(csv_path: Path, data_dir: Path, crop_col="crop_image", label_col="color"):
    df = pd.read_csv(csv_path, sep = ";")
    if crop_col not in df.columns or label_col not in df.columns:
         raise ValueError(f"CSV must contain columns `{crop_col}` and `{label_col}`")
    # filter out empty labels
    df = df[df[label_col].notna()]
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""]
    items = []
    missing = 0
    for _, row in df.iterrows():
        crop_name = row[crop_col]
        label = row[label_col]
        p = find_image(data_dir, crop_name)
        if p:
            items.append((str(p), label))
        else:
            missing += 1
    if missing:
        print(f"Warning: {missing} crop images listed in CSV were not found under {data_dir}")
    return items

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)
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

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    pbar = tqdm(loader, desc="Val  ", leave=False)
    with torch.no_grad():
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
            batch_acc = (preds == targets).float().mean().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "batch_acc": f"{batch_acc:.4f}"})
    return running_loss / total, correct / total if total else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="crops_epl_1", help="root folder containing crops (recursively)")
    p.add_argument("--csv", default="crops_index.csv", help="CSV with columns `crop_image` and `color`")
    p.add_argument("--out-dir", default="training/team_clustering/models", help="where to save best model and mapping")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    data_dir = Path(f"training/team_clustering/{args.data_dir}")
    csv_path = data_dir / args.csv
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = build_items_from_csv(csv_path, data_dir, crop_col="crop_image", label_col="color")
    if not items:
        raise SystemExit("No labeled items found. Check CSV and data directory.")

    labels = sorted({lbl for _, lbl in items})
    label2idx = {lbl: i for i, lbl in enumerate(labels)}

    train_items, val_items = train_test_split(items, test_size=args.val_split, stratify=[lbl for _, lbl in items] if len(labels) > 1 else None, random_state=42)

    # Preprocessing using MobileNetV3 weights statistics
    weights = MobileNet_V3_Small_Weights.DEFAULT

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = mobilenet_v3_small(weights=weights)
    # replace final classifier to match number of colors
    # locate last Linear in classifier
    last_lin_idx = None
    for i, m in enumerate(model.classifier):
        if isinstance(m, nn.Linear):
            last_lin_idx = i
    if last_lin_idx is None:
        raise RuntimeError("Unable to find final linear layer in MobileNetV3 classifier")
    in_features = model.classifier[last_lin_idx].in_features
    model.classifier[last_lin_idx] = nn.Linear(in_features, len(labels))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_path = out_dir / "best_mobilenetv3_small.pth"
    mapping_path = out_dir / "label2idx.json"

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs", unit="ep"):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "label2idx": label2idx}, best_path)
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(label2idx, f, ensure_ascii=False, indent=2)
            print(f"Saved best model (val_acc={best_acc:.4f}) to {best_path}")

    print("Training finished. Best val_acc:", best_acc)

if __name__ == "__main__":
    main()
