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
    """
    PyTorch Dataset for player jersey images and color labels.
    """
    def __init__(self, items, label2idx, transform=None):
        """
        Parameters
        ----------
        items : list of (image_path, label)
            List of image file paths and corresponding label strings.
        label2idx : dict
            Mapping from label strings to integer class indices.
        transform : callable, optional
            Optional torchvision transforms applied to each image.
        """
        self.items = items
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        """Return number of items in the dataset."""
        return len(self.items)

    def __getitem__(self, idx):
        """
        Load an image and its label at the given index.

        Returns
        -------
        tuple[torch.Tensor, int]
            Transformed image tensor and integer class label.
        """
        path, label = self.items[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
            if self.transform:
                img = self.transform(img)
        target = self.label2idx.get(label)
        return img, target

def find_image(data_dir: Path, crop_name: str) -> Path | None:
    """
    Locate an image file by name under a directory.

    Parameters
    ----------
    data_dir : Path
        Root directory to search.
    crop_name : str
        Image filename or relative path.

    Returns
    -------
    Path or None
        Found image path, or None if not found.
    """
    candidate = data_dir / crop_name
    if candidate.exists():
        return candidate
    # fallback: search for basename
    basename = Path(crop_name).name
    for p in data_dir.rglob(basename):
        return p
    return None

def build_items_from_csv(
        csv_path: Path,
        data_dir: Path,
        crop_col: str = "crop_image",
        label_col: str = "color"
) -> list[tuple[str, str]]:
    """
    Build a list of (image_path, label) tuples from a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    data_dir : Path
        Root directory containing crop images.
    crop_col : str
        Column name for image filenames.
    label_col : str
        Column name for class labels.

    Returns
    -------
    list of (str, str)
        List of tuples with image paths and labels.
    """
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

def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, dict | None]:
    """
    Load a MobileNetV3 model and label mapping from disk.

    Parameters
    ----------
    model_path : Path
        Path to the saved model file.
    device : torch.device
        Device to move the model to.

    Returns
    -------
    model : nn.Module
        Loaded MobileNetV3 model set to eval mode.
    label2idx : dict or None
        Mapping of label strings to class indices, or None if unavailable.
    """
    # build model skeleton and adjust final layer
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    # determine num classes from saved model or mapping
    state = torch.load(model_path, map_location="cpu")
    sd = state.get("model_state_dict", state)

    # load label2idx
    label2idx = state.get("label2idx")
    num_classes = len(label2idx)

    # replace final classifier linear
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    return model, label2idx

def extract_embeddings(model: nn.Module, imgs: torch.Tensor) -> np.ndarray:
    """
    Compute feature embeddings for a batch of images.

    Parameters
    ----------
    model : nn.Module
        Feature extractor model.
    imgs : torch.Tensor
        Batch of images on the appropriate device.

    Returns
    -------
    np.ndarray
        Array of embeddings with shape (batch_size, embedding_dim).
    """
    # imgs: tensor on device
    model.eval()
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
    """
    Project embeddings to 2D using UMAP and plot them.

    Parameters
    ----------
    embeddings : np.ndarray
        Feature embeddings of shape (N, D).
    labels : list
        Integer labels corresponding to embeddings.
    label_names : dict
        Mapping from label integers to display names.
    out_path : Path
        Path to save the resulting plot image.
    title : str
        Plot title.
    """
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_projection = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    labels_arr = np.array(labels)
    uniques = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for i, u in enumerate(uniques):
        mask = labels_arr == u
        plt.scatter(
            umap_projection[mask, 0],
            umap_projection[mask, 1],
            label=label_names[u],
            s=10,
            color=cmap(i % 20),
            alpha=0.8
        )
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()