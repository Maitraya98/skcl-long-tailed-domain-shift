"""
skcl_model.py -- two-branch SKCL model: shared encoder, contrastive head,
and two-level (fine+coarse) classifiers with prototype generation.
"""
from __future__ import annotations
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from resnet_cifar import resnet32, NormedLinear


class SKCLModelCIFAR(nn.Module):
    def __init__(self, num_fine: int, num_coarse: int, feat_dim: int = 128, use_norm: bool = True):
        super().__init__()
        self.num_fine = num_fine
        self.num_coarse = num_coarse

        self.encoder = resnet32()
        dim_in = self.encoder.feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )

        if use_norm:
            self.fc_fine = NormedLinear(dim_in, num_fine)
            self.fc_coarse = NormedLinear(dim_in, num_coarse)
        else:
            self.fc_fine = nn.Linear(dim_in, num_fine)
            self.fc_coarse = nn.Linear(dim_in, num_coarse)

        self.proto_head_fine = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        self.proto_head_coarse = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )

    def _fc_weight_as_rows(self, fc: nn.Module) -> torch.Tensor:
        if isinstance(fc, NormedLinear):
            return fc.weight.T
        return fc.weight

    def forward(self, x):
        feat = self.encoder(x)

        z = self.contrast_head(feat)
        z_bar = F.normalize(z, dim=1)

        logits_fine = self.fc_fine(feat)
        logits_coarse = self.fc_coarse(feat)

        proto_fine = F.normalize(self.proto_head_fine(self._fc_weight_as_rows(self.fc_fine)), dim=1)
        proto_coarse = F.normalize(self.proto_head_coarse(self._fc_weight_as_rows(self.fc_coarse)), dim=1)
        prototypes = torch.cat([proto_fine, proto_coarse], dim=0)

        return z_bar, logits_fine, logits_coarse, prototypes


if __name__ == "__main__":
    m = SKCLModelCIFAR(num_fine=100, num_coarse=20, feat_dim=128)
    x = torch.randn(8, 3, 32, 32)
    z_bar, logits_fine, logits_coarse, prototypes = m(x)
    print("z_bar:", z_bar.shape, "logits_fine:", logits_fine.shape,
          "logits_coarse:", logits_coarse.shape, "prototypes:", prototypes.shape)
    assert z_bar.shape == (8, 128)
    assert logits_fine.shape == (8, 100)
    assert logits_coarse.shape == (8, 20)
    assert prototypes.shape == (120, 128)
    assert torch.allclose(z_bar.norm(dim=1), torch.ones(8), atol=1e-4)
    assert torch.allclose(prototypes.norm(dim=1), torch.ones(120), atol=1e-4)
    print("OK: shapes match paper's two-level framework")
