"""

Architecture:
    Two EfficientNet-B4 encoders (one for EO/RGB, one for SAR/1-ch) extract
    multi-scale features. At each decoder stage the feature maps from both
    encoders are concatenated (early fusion) before a lightweight DoubleConv
    fusion block, then decoded with transposed convolutions back to the full
    256×256 resolution.

Why EfficientNet-B4?
    Strong ImageNet-pretrained features for the EO branch while remaining
    light enough for edge/cloud inference. The SAR encoder is initialised
    randomly (no pretrained weights exist for single-channel SAR).
"""

import torch
import torch.nn as nn
import timm


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DualEncoderUNet(nn.Module):
    """
    Dual-encoder UNet with EfficientNet-B4 backbones.

    Input:
        eo  — (B, 3, H, W)  pre-event Electro-Optical image
        sar — (B, 1, H, W)  post-event Synthetic Aperture Radar image

    Output:
        logits — (B, 1, H, W)  raw (un-sigmoided) change probability map
    """

    # EfficientNet-B4 multi-scale channel widths at out_indices (1,2,3,4)
    _EFF_CHANNELS = [32, 56, 160, 448]

    def __init__(self, pretrained: bool = True):
        super().__init__()

        #EO Encoder: 3-channel, ImageNet pretrained 
        self.eo_encoder = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        #SAR Encoder: 1-channel
        self.sar_encoder = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        # Replace the first conv to accept single-channel SAR
        self.sar_encoder.conv_stem = nn.Conv2d(
            1, 48, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Multi-scale fusion (concat EO + SAR → fuse) 
        # Channels after concat = 2 × EfficientNet-B4 channel width
        self.fuse4 = DoubleConv(self._EFF_CHANNELS[3] * 2, 256)   # 896  → 256
        self.fuse3 = DoubleConv(self._EFF_CHANNELS[2] * 2, 128)   # 320  → 128
        self.fuse2 = DoubleConv(self._EFF_CHANNELS[1] * 2,  64)   # 112  → 64
        self.fuse1 = DoubleConv(self._EFF_CHANNELS[0] * 2,  32)   # 64   → 32

        # Decoder 
        self.up4  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256, 128)   # 128 up + 128 skip

        self.up3  = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128,  64)   # 64  up + 64  skip

        self.up2  = nn.ConvTranspose2d(64,   32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64,   32)   # 32  up + 32  skip

        self.up1  = nn.ConvTranspose2d(32,   16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(16,   16)

        # Extra upsample to recover full 256×256 resolution from 8×8 stem
        self.up0  = nn.ConvTranspose2d(16,    8, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(8,     8)

        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, eo: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        # Encode 
        eo_feats  = self.eo_encoder(eo)    # list of 4 feature maps
        sar_feats = self.sar_encoder(sar)

        
        f4 = self.fuse4(torch.cat([eo_feats[3], sar_feats[3]], dim=1))
        f3 = self.fuse3(torch.cat([eo_feats[2], sar_feats[2]], dim=1))
        f2 = self.fuse2(torch.cat([eo_feats[1], sar_feats[1]], dim=1))
        f1 = self.fuse1(torch.cat([eo_feats[0], sar_feats[0]], dim=1))

        #Decode
        x = self.up4(f4)
        x = self.dec4(torch.cat([x, f3], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, f2], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, f1], dim=1))

        x = self.up1(x)
        x = self.dec1(x)

        x = self.up0(x)
        x = self.dec0(x)

        return self.final(x)   # (B, 1, H, W) — raw logits