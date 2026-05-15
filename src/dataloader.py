"""

EO-SAR Binary Change Detection 
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class ChangeDetectionDataset(Dataset):
    """
    Loads co-registered EO (pre-event, RGB) and SAR (post-event, grayscale) image pairs
    with binary change masks.

    Label remapping (applied here before any computation):
        Original 0 (Background) → 0 (No-Change)
        Original 1 (Intact)     → 0 (No-Change)
        Original 2 (Damaged)    → 1 (Change)
        Original 3 (Destroyed)  → 1 (Change)

    
            
    """

    def __init__(self, root_dir: str, split: str = "train", patch_size: int = 256, augment: bool = False):
        self.root_dir   = root_dir
        self.split      = split
        self.patch_size = patch_size
        self.augment    = augment

        self.pre_dir    = os.path.join(root_dir, "pre-event")
        self.post_dir   = os.path.join(root_dir, "post-event")
        self.target_dir = os.path.join(root_dir, "target")

        self.files = sorted(os.listdir(self.pre_dir))
        print(f"[{split}] Found {len(self.files)} samples in {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        fname = self.files[idx]

        pre    = np.array(Image.open(os.path.join(self.pre_dir,    fname)))   # H×W×3
        post   = np.array(Image.open(os.path.join(self.post_dir,   fname)))   # H×W or H×W×1
        target = np.array(Image.open(os.path.join(self.target_dir, fname)))   # H×W, values 0-3

        # ── Label remapping: {0,1} → 0 (no-change), {2,3} → 1 (change) ──
        target = np.where(target >= 2, 1, 0).astype(np.float32)

        # Ensure SAR has a channel dim
        if post.ndim == 2:
            post = post[:, :, np.newaxis]                                     # H×W×1

        # Nodata mask: EO pixels that are entirely black (sensor fill value)
        nodata_mask = (pre.sum(axis=2) == 0).astype(np.float32)              # H×W

        # Normalise to [0, 1]
        pre  = pre.astype(np.float32)  / 255.0
        post = post.astype(np.float32) / 255.0

        # Patch extraction
        if self.split == "train":
            pre, post, target, nodata_mask = self._random_patch(pre, post, target, nodata_mask)
        else:
            pre, post, target, nodata_mask = self._center_patch(pre, post, target, nodata_mask)

        # Augmentation (train only)
        if self.augment and self.split == "train":
            pre, post, target, nodata_mask = self._augment(pre, post, target, nodata_mask)

        # NumPy → Tensor
        pre    = torch.from_numpy(pre).permute(2, 0, 1)              # 3×H×W
        post   = torch.from_numpy(post).permute(2, 0, 1)             # 1×H×W
        target = torch.from_numpy(target.astype(np.float32))         # H×W
        nodata = torch.from_numpy(nodata_mask)                        # H×W

        return {"pre": pre, "post": post, "target": target, "nodata": nodata, "fname": fname}

    

    def _random_patch(self, pre, post, target, nodata):
        H, W = pre.shape[:2]
        p    = self.patch_size
        top  = random.randint(0, H - p)
        left = random.randint(0, W - p)
        return (
            pre   [top:top+p, left:left+p],
            post  [top:top+p, left:left+p],
            target[top:top+p, left:left+p],
            nodata[top:top+p, left:left+p],
        )

    def _center_patch(self, pre, post, target, nodata):
        H, W = pre.shape[:2]
        p    = self.patch_size
        top  = (H - p) // 2
        left = (W - p) // 2
        return (
            pre   [top:top+p, left:left+p],
            post  [top:top+p, left:left+p],
            target[top:top+p, left:left+p],
            nodata[top:top+p, left:left+p],
        )

    def _augment(self, pre, post, target, nodata):
        """Spatial augmentations applied identically to all four arrays."""
        if random.random() > 0.5:
            pre, post, target, nodata = [np.fliplr(x).copy() for x in [pre, post, target, nodata]]
        if random.random() > 0.5:
            pre, post, target, nodata = [np.flipud(x).copy() for x in [pre, post, target, nodata]]
        if random.random() > 0.5:
            k = random.randint(1, 3)
            pre, post, target, nodata = [np.rot90(x, k).copy() for x in [pre, post, target, nodata]]
        return pre, post, target, nodata


# ── DataLoader factory ─

def get_dataloaders(
    train_dir:   str,
    val_dir:     str  = None,
    batch_size:  int  = 8,
    patch_size:  int  = 256,
    num_workers: int  = 2,
) -> tuple:
    """
    Returns (train_loader, val_loader). val_loader is None if val_dir is not given.
    """
    train_ds = ChangeDetectionDataset(train_dir, split="train", patch_size=patch_size, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = None
    if val_dir:
        val_ds = ChangeDetectionDataset(val_dir, split="val", patch_size=patch_size, augment=False)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader
