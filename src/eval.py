"""


Usage:
    python eval.py --data_path /path/to/test --weights /path/to/best_model.pth

Optionally saves qualitative prediction visualisations with --save_viz.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import ChangeDetectionDataset
from model   import DualEncoderUNet


# ── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()

    total_TP = total_TN = total_FP = total_FN = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            eo     = batch["pre"].to(device)
            sar    = batch["post"].to(device)
            target = batch["target"].unsqueeze(1).to(device)
            nodata = batch["nodata"].unsqueeze(1).to(device)

            pred = torch.sigmoid(model(eo, sar))
            pred_bin = (pred > threshold).float()

            valid    = (nodata == 0)
            pred_v   = pred_bin[valid]
            target_v = target[valid]

            total_TP += ((pred_v == 1) & (target_v == 1)).sum().item()
            total_TN += ((pred_v == 0) & (target_v == 0)).sum().item()
            total_FP += ((pred_v == 1) & (target_v == 0)).sum().item()
            total_FN += ((pred_v == 0) & (target_v == 1)).sum().item()

    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall    = total_TP / (total_TP + total_FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = total_TP / (total_TP + total_FP + total_FN + 1e-8)
    accuracy  = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + 1e-8)

    return {
        "iou"       : iou,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "accuracy"  : accuracy,
        "TP"        : total_TP,
        "TN"        : total_TN,
        "FP"        : total_FP,
        "FN"        : total_FN,
    }


# ── Qualitative visualisation ────────────────────────────────────────────────

def save_visualisations(model, loader, device, out_dir: str, n_samples: int = 8,
                        threshold: float = 0.5):
    """Save side-by-side (EO | GT | Prediction) panels for up to n_samples images."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in loader:
            eo     = batch["pre"].to(device)
            sar    = batch["post"].to(device)
            target = batch["target"]
            fnames = batch["fname"]

            pred = torch.sigmoid(model(eo, sar))
            pred_bin = (pred > threshold).float().squeeze(1).cpu().numpy()

            for i in range(eo.size(0)):
                eo_img  = eo[i].cpu().permute(1, 2, 0).numpy()
                gt      = target[i].cpu().numpy()
                pr      = pred_bin[i]
                fname   = fnames[i]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(np.clip(eo_img, 0, 1))
                axes[0].set_title("Pre-event (EO)")
                axes[0].axis("off")

                axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pr, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                plt.suptitle(fname, fontsize=9)
                plt.tight_layout()
                save_path = os.path.join(out_dir, f"viz_{os.path.splitext(fname)[0]}.png")
                plt.savefig(save_path, dpi=120, bbox_inches="tight")
                plt.close(fig)

                saved += 1
                if saved >= n_samples:
                    return


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_results(metrics: dict, split_name: str = "Test"):
    print(f"\n{'='*55}")
    print(f"  {split_name} Results (Change class, label = 1)")
    print(f"{'='*55}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"\n  Confusion Matrix (pixels):")
    print(f"                Pred=0      Pred=1")
    print(f"  Actual=0   {metrics['TN']:>10,}  {metrics['FP']:>10,}   (True Neg / False Pos)")
    print(f"  Actual=1   {metrics['FN']:>10,}  {metrics['TP']:>10,}   (False Neg / True Pos)")
    print(f"{'='*55}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DualEncoderUNet on a change detection split")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root directory of the split to evaluate (must contain pre-event/, post-event/, target/)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="test",
                        help="Split label used for logging (default: test)")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Patch size (should match training, default: 256)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation (default: 4)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary classification threshold (default: 0.5)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_viz", action="store_true",
                        help="Save qualitative prediction visualisations")
    parser.add_argument("--viz_dir", type=str, default="visualisations",
                        help="Directory to save visualisations (default: visualisations/)")
    parser.add_argument("--n_viz", type=int, default=8,
                        help="Number of visualisation samples (default: 8)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")
    print(f"Data    : {args.data_path}")

    # Dataset & loader
    dataset = ChangeDetectionDataset(
        root_dir   = args.data_path,
        split      = args.split,
        patch_size = args.patch_size,
        augment    = False,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # Model
    model = DualEncoderUNet(pretrained=False).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print("Model loaded successfully.\n")

    # Evaluate
    metrics = evaluate(model, loader, device, threshold=args.threshold)
    print_results(metrics, split_name=args.split.capitalize())

    # Optional visualisations
    if args.save_viz:
        save_visualisations(
            model, loader, device,
            out_dir   = args.viz_dir,
            n_samples = args.n_viz,
            threshold = args.threshold,
        )
        print(f"Visualisations saved to: {args.viz_dir}/")


if __name__ == "__main__":
    main()