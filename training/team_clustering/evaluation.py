import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from training.team_clustering.utils import (JerseyDataset, resolve_csv, build_items_from_csv, load_model,
                                            extract_embeddings, plot_umap)

from training.logger import LOGGER

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="crops_index.csv", help="CSV with columns `crop_image` and `color` (path or name)")
parser.add_argument("--test-dir", default="test-crops", help="test folder path or name (relative to CSV folder if not absolute)")
parser.add_argument("--model", default="training/team_clustering/models/best_mobilenetv3_small.pth", help="trained model .pth")
parser.add_argument("--mapping", default="training/team_clustering/models/label2idx.json", help="label2idx json")
parser.add_argument("--out-dir", default="training/team_clustering/eval", help="where to save plots and reports")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--img-size", type=int, default=224)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--device", default=None, help="device override (cuda/mps/cpu)")
args = parser.parse_args()

# resolve CSV path (searching project if needed)
csv_path = Path(args.csv)
csv_found = resolve_csv(csv_path, csv_path.parent)
if csv_found is None:
    raise SystemExit(f"CSV not found: {csv_path}")
csv_path = csv_found

# images root is the folder containing the CSV
images_root = csv_path.parent

# resolve test dir: prefer explicit path, fallback to folder relative to CSV
test_dir = Path(args.test_dir)
if not test_dir.exists():
    alt = images_root / test_dir
    if alt.exists():
        test_dir = alt
    else:
        LOGGER.warning(f"Warning: test dir ` {args.test_dir} ` not found (neither absolute nor relative to CSV). Will fallback to images referenced in CSV")
        test_dir = None

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# build items using images_root as search base
items = build_items_from_csv(csv_path, images_root, crop_col="crop_image", label_col="color")

# filter to test items if test_dir was resolved
if test_dir is not None and test_dir.exists():
    if hasattr(Path, "is_relative_to"):
        def _is_in_test(p):
            return Path(p).resolve().is_relative_to(test_dir.resolve())
    else:
        def _is_in_test(p):
            return str(Path(p).resolve()).startswith(str(test_dir.resolve()))
    test_items = [(p, lbl) for p, lbl in items if _is_in_test(p)]

    # fallback: include items whose path contains test-dir name
    if not test_items:
        test_items = [(p, lbl) for p, lbl in items if str(test_dir.name) in str(Path(p).parts)]
else:
    # if the test dir is not available, treat all items as test
    test_items = items

if not test_items:
    raise SystemExit("No test items found. Check CSV and test directory.")

# prepare label mapping
mapping_path = Path(args.mapping)
device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))

model_path = Path(args.model)
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model, label2idx = load_model(model_path, mapping_path, device)
idx2label = {v: k for k, v in label2idx.items()}

# transforms using model weights stats
transform = transforms.Compose([
    transforms.Resize(int(args.img_size * 256 / 224)),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
])

test_ds = JerseyDataset(test_items, label2idx, transform=transform)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

all_preds = []
all_targets = []
all_embeddings = []

pbar = tqdm(test_loader, desc="Eval")
for imgs, targets in pbar:
    imgs = imgs.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        outputs = model(imgs)
        _, preds = outputs.max(1)
        emb = extract_embeddings(model, imgs)
    all_preds.extend(preds.cpu().tolist())
    all_targets.extend(targets.cpu().tolist())
    all_embeddings.append(emb)
    batch_acc = (preds == targets).float().mean().item()
    pbar.set_postfix({"batch_acc": f"{batch_acc:.4f}"})

all_embeddings = np.vstack(all_embeddings)
acc = accuracy_score(all_targets, all_preds)
report = classification_report(all_targets, all_preds, target_names=[idx2label[i] for i in np.unique(all_preds)])
cm = confusion_matrix(all_targets, all_preds)

# save results
results_df = pd.DataFrame({
    "target_idx": all_targets,
    "pred_idx": all_preds,
    "target_label": [idx2label[i] for i in all_targets],
    "pred_label": [idx2label[i] for i in all_preds],
})
results_df.to_csv(out_dir / "eval_results.csv", index=False)
with open(out_dir / "eval_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(np.array2string(cm))

# plot umap / tsne with ground truth
plot_path = out_dir / "umap_ground_truth.png"
plot_umap(all_embeddings, all_targets, idx2label, plot_path, title=f"UMAP ground truth (acc={acc:.4f})")

LOGGER.info(f"Done. Accuracy: {acc:.4f}. Results saved to ` {out_dir} `")
LOGGER.info(f"UMAP plot: ` {plot_path} `")
