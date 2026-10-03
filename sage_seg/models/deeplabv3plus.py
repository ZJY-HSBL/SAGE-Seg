from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101, ResNet101_Weights


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = super().forward(x)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates=(6, 12, 18)):
        super().__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        branches.extend(ASPPConv(in_channels, out_channels, r) for r in rates)
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, output_stride: int = 16):
        super().__init__()
        if output_stride == 16:
            dilation = [False, False, True]
        elif output_stride == 8:
            dilation = [False, True, True]
        else:
            raise ValueError("output_stride must be 8 or 16")
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet101(weights=weights, replace_stride_with_dilation=dilation)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)
        return low, high


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        output_stride: int = 16,
        aspp_channels: int = 256,
    ):
        super().__init__()
        self.backbone = ResNet101Backbone(pretrained_backbone, output_stride)
        rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(2048, aspp_channels, rates)
        self.low_project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        low, high = self.backbone(x)
        high = self.aspp(high)
        low = self.low_project(low)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([high, low], dim=1))
        x = self.classifier(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

    def parameter_groups(self, base_lr: float, head_lr_multiplier: float = 10.0):
        backbone_params = list(self.backbone.parameters())
        head_modules = [self.aspp, self.low_project, self.decoder, self.classifier]
        head_params = [p for m in head_modules for p in m.parameters()]
        return [
            {"params": backbone_params, "lr": base_lr, "initial_lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_multiplier,
             "initial_lr": base_lr * head_lr_multiplier},
        ]
