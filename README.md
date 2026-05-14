# EO-SAR Binary Change Detection with Dual Encoder EfficientNetB4+UNet

PyTorch implementation of a dual-encoder U-Net for binary change detection using
pre-event electro-optical (EO/RGB) imagery and post-event SAR/grayscale imagery.

The model predicts a pixel-wise binary change mask:

- `0`: no change
- `1`: change

Original target labels are remapped in the dataloader:

- `0` background -> `0` no change
- `1` intact -> `0` no change
- `2` damaged -> `1` change
- `3` destroyed -> `1` change

## Repository Structure

```text
galaxeye-change-detection/
├── config.yaml
├── README.md
├── requirements.txt
└── src/
    ├── dataloader.py
    ├── eval.py
    ├── model.py
    └── train.py
```

## Dataset Layout

Each split should contain three folders with matching filenames:

```text
data/
├── train/
│   ├── pre-event/    # RGB EO images
│   ├── post-event/   # grayscale SAR images
│   └── target/       # masks with labels 0-3
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

The dataloader normalizes image values to `[0, 1]`, converts SAR images to
single-channel tensors, masks EO no-data pixels where RGB is all zeros, and
extracts patches. Training uses random patches and augmentations; validation
and testing use center patches.

## Model

`DualEncoderUNet` uses two EfficientNet-B4 encoders:

- EO branch: 3-channel RGB input, optionally ImageNet-pretrained
- SAR branch: 1-channel SAR input, randomly initialized

Multi-scale EO and SAR features are concatenated, fused with convolution blocks,
and decoded back to a full-resolution binary change map.

Input tensors:

```text
EO:  (B, 3, H, W)
SAR: (B, 1, H, W)
```

Output:

```text
logits: (B, 1, H, W)
```

## Installation

Create an environment and install the required packages:

```bash
pip install torch torchvision timm numpy pillow pyyaml tqdm matplotlib
```



## Configuration

Training settings are stored in `config.yaml`:

```yaml
train_dir: "data/train"
val_dir: "data/val"
epochs: 30
seed: 42
pretrained: true
batch_size: 8
patch_size: 256
num_workers: 2
learning_rate: 0.0001
weight_decay: 0.01
scheduler_patience: 3
scheduler_factor: 0.5
pos_weight: 62.7
checkpoint_dir: "checkpoints"
```

Update `train_dir` and `val_dir` before training.



## Training

From the project root:

```bash
cd galaxeye-change-detection
python src/train.py --config config.yaml
```

The training script:

- builds train and validation dataloaders
- trains `DualEncoderUNet`
- uses AdamW optimization
- uses a Dice loss plus weighted BCE with logits
- tracks IoU, precision, recall, and F1
- reduces learning rate on validation IoU plateau
- saves the best validation IoU checkpoint to:

```text
checkpoints:
https://drive.google.com/file/d/1xptDdhvmItdiEapyfQ34qO1l8WRoHclL/view?usp=sharing
```

## Evaluation

Evaluated a trained checkpoint on a split of:

```bash
python src/eval.py \
  --data_path data/test \
  --weights checkpoints/best_model.pth \
  --split test \
  --patch_size 256 \
  --batch_size 4 \
  --threshold 0.5
```

The evaluator reports:

- IoU
- precision
- recall
- F1
- accuracy


## Save Visualizations

To save qualitative prediction panels:

```bash
python src/eval.py \
  --data_path data/test \
  --weights checkpoints/best_model.pth \
  --save_viz \
  --viz_dir visualisations \
  --n_viz 8
```

Each visualization contains:

```text
Pre-event EO | Ground Truth | Prediction
```

## Notes

- `pos_weight` is set for strong class imbalance, where changed pixels are a
  small fraction of the dataset.
- Validation and test behavior use center crops, so `patch_size` should match
  the size used during training.
- The EO no-data mask excludes all-black EO pixels from loss and metric
  calculations.
  - Also made use of Google Colab for Understanding Data Better, Have attached the Colab Notebooks too

