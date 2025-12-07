import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from training.logger import LOGGER

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
        with Image.open(path) as im:
            img = im.convert("RGB")
            if self.transform:
                img = self.transform(img)
        target = self.label2idx.get(label)
        return img, target

def find_image(data_dir: Path, crop_name: str) -> Path | None:
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

def build_items_from_csv(
        csv_path: Path,
        data_dir: Path,
        crop_col: str = "crop_image",
        label_col: str = "color"
) -> list[tuple[str, str]]:
    df = pd.read_csv(csv_path)
    if crop_col not in df.columns or label_col not in df.columns:
         raise ValueError(f"CSV must contain columns `{crop_col}` and `{label_col}`")
    # filter out empty labels
    df = df[df[label_col].notna()]
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""]
    items = []
    missing_count = 0
    for _, row in df.iterrows():
        crop_name = row[crop_col]
        label = row[label_col]
        p = find_image(data_dir, crop_name)
        if p:
            items.append((str(p), label))
        else:
            missing_count += 1
    if missing_count:
        LOGGER.warning(f"Warning: {missing_count} crop images listed in CSV were not found under {data_dir}")
    return items

def load_model(model_path: Path, mapping_path: Path, device: torch.device) -> tuple[nn.Module, dict | None]:
    # load label2idx
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            label2idx = json.load(f)
    else:
        label2idx = None

    # build model skeleton and adjust final layer
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    # determine num classes from saved model or mapping
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state
    # infer num_classes from state dict final linear weight shape if possible
    final_w_keys = [k for k in sd.keys() if "classifier" in k and "weight" in k]
    num_classes = None
    if final_w_keys:
        # take last classifier.weight
        key = sorted(final_w_keys)[-1]
        num_classes = sd[key].shape[0]
    if label2idx is not None:
        num_classes = len(label2idx)
    if num_classes is None:
        raise RuntimeError("Unable to determine number of classes from mapping or model")

    # replace final classifier linear
    last_lin_idx = None
    for i, m in enumerate(model.classifier):
        if isinstance(m, nn.Linear):
            last_lin_idx = i
    if last_lin_idx is None:
        raise RuntimeError("Unable to find final linear in MobileNetV3 classifier")
    in_features = model.classifier[last_lin_idx].in_features
    model.classifier[last_lin_idx] = nn.Linear(in_features, num_classes)

    model.load_state_dict(sd if "model_state_dict" not in state else state["model_state_dict"])
    model.to(device)
    model.eval()

    if label2idx is None and "label2idx" in state:
        label2idx = state["label2idx"]
    if isinstance(label2idx, dict):
        # ensure keys are strings and values ints
        label2idx = {str(k): int(v) for k, v in label2idx.items()}
    return model, label2idx

def extract_embeddings(model: nn.Module, imgs: torch.Tensor) -> np.ndarray:
    # imgs: tensor on device
    with torch.no_grad():
        x = model.features(imgs)
        # model.avgpool usually exists
        if hasattr(model, "avgpool"):
            x = model.avgpool(x)
        else:
            # fallback adaptive pool to (1,1)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        # apply classifier except last linear to get embedding
        if isinstance(model.classifier, torch.nn.Sequential) and len(model.classifier) > 1:
            feat = model.classifier[:-1](x)
        else:
            feat = x
    return feat.cpu().numpy()

def plot_umap(
        embeddings: np.ndarray,
        labels: list,
        label_names: dict,
        out_path: Path,
        title: str = "UMAP ground truth"
) -> None:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X2 = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    uniques = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for i, u in enumerate(uniques):
        idxs = [j for j, lab in enumerate(labels) if lab == u]
        plt.scatter(X2[idxs, 0], X2[idxs, 1], label=label_names[u], s=10, color=cmap(i % 20), alpha=0.8)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()