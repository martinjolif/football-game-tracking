from src.team_clustering.utils import load_model, InferenceJerseyDataset, extract_embeddings
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np


def classify_player_from_image(crop_images, model_path: str, img_size: int, batch_size: int, device: str):
    """
    Classify the team of a player from a cropped image.

    Args:
        crop_image (PIL.Image): Cropped image of the player.
    """

    model, label2idx = load_model(Path(model_path), device)

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    test_ds = InferenceJerseyDataset(crop_images, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    for imgs in tqdm(test_loader, desc="Eval"):
        imgs = imgs.to(device)
        with torch.no_grad():
            emb = extract_embeddings(model, imgs)
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)