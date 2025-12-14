from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import Dataset
from PIL import Image

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

class InferenceJerseyDataset(Dataset):
    """
    PyTorch Dataset for player jersey images and color labels.
    """
    def __init__(self, images, transform=None):
        """
        Parameters
        ----------
        transform : callable, optional
            Optional torchvision transforms applied to each image.
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        """Return number of items in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load an image and its label at the given index.

        Returns
        -------
        tuple[torch.Tensor, int]
            Transformed image tensor and integer class label.
        """
        return self.transform(self.images[idx]) if self.transform else self.images[idx]