"""

The script reads all hyperparameters from the YAML config, trains a
DualEncoderUNet, and saves the best checkpoint (by Val IoU) to the
path specified in config['checkpoint_dir'].
"""

import os
import argparse
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataloader import get_dataloaders
from model   import DualEncoderUNet




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Loss functions 

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred   = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def bce_loss(pred: torch.Tensor, target: torch.Tensor, nodata: torch.Tensor,
             pos_weight_val: float) -> torch.Tensor:
    """Weighted BCE with logits, restricted to valid (non-nodata) pixels."""
    valid        = (nodata == 0)
    pred_valid   = pred[valid]
    target_valid = target[valid]
    if pred_valid.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    pw        = torch.tensor([pos_weight_val], device=pred.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    return criterion(pred_valid, target_valid)


def combined_loss(pred: torch.Tensor, target: torch.Tensor, nodata: torch.Tensor,
                  pos_weight_val: float) -> torch.Tensor:
    valid          = (nodata == 0).unsqueeze(1)
    pred_masked    = pred   * valid
    target_masked  = target.unsqueeze(1) * valid
    return dice_loss(pred_masked, target_masked) + bce_loss(
        pred.squeeze(1), target, nodata, pos_weight_val
    )


# Metrics 
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, nodata: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    pred_prob = torch.sigmoid(pred).squeeze(1)
    valid     = (nodata == 0)

    pred_bin = (pred_prob > threshold).float()
    pred_v   = pred_bin[valid]
    target_v = target[valid]

    tp = (pred_v * target_v).sum().item()
    fp = (pred_v * (1 - target_v)).sum().item()
    fn = ((1 - pred_v) * target_v).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


# Single epoch 
def train_one_epoch(model, loader, optimizer, device, pos_weight_val):
    model.train()
    total_loss  = 0.0
    all_metrics = {"iou": [], "precision": [], "recall": [], "f1": []}

    for batch in tqdm(loader, desc="  Train"):
        eo     = batch["pre"].to(device)
        sar    = batch["post"].to(device)
        target = batch["target"].to(device)
        nodata = batch["nodata"].to(device)

        optimizer.zero_grad()
        pred = model(eo, sar)
        loss = combined_loss(pred, target, nodata, pos_weight_val)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in compute_metrics(pred.detach(), target, nodata).items():
            all_metrics[k].append(v)

    return total_loss / len(loader), {k: float(np.mean(v)) for k, v in all_metrics.items()}


def validate(model, loader, device, pos_weight_val):
    model.eval()
    total_loss  = 0.0
    all_metrics = {"iou": [], "precision": [], "recall": [], "f1": []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Val  "):
            eo     = batch["pre"].to(device)
            sar    = batch["post"].to(device)
            target = batch["target"].to(device)
            nodata = batch["nodata"].to(device)

            pred = model(eo, sar)
            loss = combined_loss(pred, target, nodata, pos_weight_val)

            total_loss += loss.item()
            for k, v in compute_metrics(pred, target, nodata).items():
                all_metrics[k].append(v)

    return total_loss / len(loader), {k: float(np.mean(v)) for k, v in all_metrics.items()}


#Main training loop 

def main(config: dict):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_dir   = config["train_dir"],
        val_dir     = config.get("val_dir"),
        batch_size  = config["batch_size"],
        patch_size  = config["patch_size"],
        num_workers = config.get("num_workers", 2),
    )

    # Model
    model = DualEncoderUNet(pretrained=config.get("pretrained", True)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimiser & scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",
        patience = config["scheduler_patience"],
        factor   = 0.5,
        verbose  = True,
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_val_iou = 0.0

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 55)

        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, device, config["pos_weight"]
        )
        print(
            f"  Train → loss:{tr_loss:.4f}  IoU:{tr_m['iou']:.4f}  "
            f"F1:{tr_m['f1']:.4f}  P:{tr_m['precision']:.4f}  R:{tr_m['recall']:.4f}"
        )

        if val_loader:
            va_loss, va_m = validate(model, val_loader, device, config["pos_weight"])
            print(
                f"  Val   → loss:{va_loss:.4f}  IoU:{va_m['iou']:.4f}  "
                f"F1:{va_m['f1']:.4f}  P:{va_m['precision']:.4f}  R:{va_m['recall']:.4f}"
            )
            scheduler.step(va_m["iou"])

            if va_m["iou"] > best_val_iou:
                best_val_iou = va_m["iou"]
                ckpt_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ Saved best model (Val IoU: {best_val_iou:.4f}) → {ckpt_path}")

    print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualEncoderUNet for change detection")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)