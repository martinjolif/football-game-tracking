from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from training.logger import LOGGER
from training.team_clustering.utils import JerseyDataset, build_items_from_csv, plot_umap
from src.team_clustering.utils import load_model, extract_embeddings

COLOR_LABELS = ["blue", "red", "black", "white", "yellow", "green", "orange", "purple", "pink", "gray"]

def evaluate_model(
        model_path: Path,
        csv_path: Path,
        test_dir: str,
        out_dir: Path,
        batch_size: int = 64,
        img_size: int = 224,
        device: torch.device = None
) -> np.ndarray:
    """
    Evaluate a model on a test dataset and save metrics, predictions, and UMAP plot.
    Returns embeddings.
    """
    test_dir = Path(test_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build items
    items = build_items_from_csv(csv_path, test_dir, crop_col="crop_image", label_col="color")
    if test_dir.exists():
        test_items = [(p, lbl) for p, lbl in items if Path(p).resolve().is_relative_to(test_dir.resolve())]
        if not test_items:
            test_items = [(p, lbl) for p, lbl in items if test_dir.name in str(Path(p).parts)]
    else:
        test_items = items
    if not test_items:
        raise ValueError("No test items found. Check CSV and test directory.")

    # Device
    device = torch.device(device) if device else \
        torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # Load model
    model, label2idx = load_model(Path(model_path), device)
    # Add missing labels dynamically
    for lbl in COLOR_LABELS:
        if lbl not in label2idx:
            label2idx[lbl] = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    test_ds = JerseyDataset(test_items, label2idx, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Evaluation
    all_preds, all_targets, all_embeddings = [], [], []
    for imgs, targets in tqdm(test_loader, desc="Eval"):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            preds = outputs.argmax(1)
            emb = extract_embeddings(model, imgs)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)

    # Save results
    pd.DataFrame({
        "target_idx": all_targets, "pred_idx": all_preds,
        "target_label": [idx2label[i] for i in all_targets],
        "pred_label": [idx2label[i] for i in all_preds],
    }).to_csv(out_dir / "test_set_results.csv", index=False)

    plot_path = out_dir / "umap_test_set.png"
    plot_umap(all_embeddings, all_targets, idx2label, plot_path, title=f"UMAP on the test set")

    LOGGER.info(f"Done. Results saved to `{out_dir}`")
    LOGGER.info(f"UMAP plot: `{plot_path}`")

    return all_embeddings