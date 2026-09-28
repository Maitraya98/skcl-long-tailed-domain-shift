"""
resnet_cifar.py

The public BCL repo only ships ImageNet-scale ResNet50/ResNeXt50. This is
the standard CIFAR ResNet-32 (He et al. 2016, 6n+2 layers, n=5) that
Table 1/2 of the paper actually use, wrapped in BCLModelCIFAR mirroring
the original BCLModel's 3-output interface so BalSCL/LogitAdjust work
unmodified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockCIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                pad = planes - in_planes
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2], (0, 0, 0, 0, pad // 2, pad - pad // 2), "constant", 0
                    )
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, block=BasicBlockCIFAR, num_blocks=(5, 5, 5)):
        super().__init__()
        self.in_planes = 16
        self.feat_dim = 64

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


def resnet32(**kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlockCIFAR, [5, 5, 5], **kwargs)


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        return self.s * F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class BCLModelCIFAR(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 128, use_norm: bool = True):
        super().__init__()
        self.encoder = resnet32()
        dim_in = self.encoder.feat_dim

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        self.head_fc = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        self.fc = NormedLinear(dim_in, num_classes) if use_norm else nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat)
        centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)
        return feat_mlp, logits, centers_logits


if __name__ == "__main__":
    m = BCLModelCIFAR(num_classes=100)
    n_conv_layers = sum(1 for mod in m.encoder.modules() if isinstance(mod, nn.Conv2d))
    print(f"Conv2d layers in encoder: {n_conv_layers} (expect 31, +1 fc = 32 total)")

    x = torch.randn(8, 3, 32, 32)
    feat_mlp, logits, centers = m(x)
    print("feat_mlp:", feat_mlp.shape, "logits:", logits.shape, "centers:", centers.shape)
    assert feat_mlp.shape == (8, 128)
    assert logits.shape == (8, 100)
    assert centers.shape == (100, 128)
    print("OK: shapes match expected BCLModel interface")
