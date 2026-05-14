import torch
import torch.nn as nn
import torchvision.models as models


class SpatialChannelAttention(nn.Module):
    """A lightweight RINet-style spatial and channel attention block."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.spatial(x) * self.channel(x)


class TinyEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
            except AttributeError:
                backbone = models.resnet18(pretrained=True)
        else:
            try:
                backbone = models.resnet18(weights=None)
            except TypeError:
                backbone = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.out_channels = 512

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def make_encoder(backbone: str, pretrained: bool = False) -> nn.Module:
    if backbone == "tiny":
        return TinyEncoder()
    if backbone == "resnet18":
        return ResNet18Encoder(pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {backbone}")


class MultiBranchFIQANet(nn.Module):
    def __init__(
        self,
        backbone: str = "tiny",
        use_left_eye: bool = True,
        use_right_eye: bool = True,
        use_mouth: bool = False,
        use_attention: bool = True,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_left_eye = use_left_eye
        self.use_right_eye = use_right_eye
        self.use_mouth = use_mouth
        self.use_attention = use_attention
        self.global_encoder = make_encoder(backbone, pretrained=pretrained)
        self.left_encoder = make_encoder(backbone, pretrained=pretrained) if use_left_eye else None
        self.right_encoder = make_encoder(backbone, pretrained=pretrained) if use_right_eye else None
        self.mouth_encoder = make_encoder(backbone, pretrained=pretrained) if use_mouth else None

        channels = self.global_encoder.out_channels
        self.rinet_attention = SpatialChannelAttention(channels) if use_attention else None
        self.pool = nn.AdaptiveAvgPool2d(1)

        branch_count = 1
        branch_count += int(use_left_eye)
        branch_count += int(use_right_eye)
        branch_count += int(use_mouth)
        branch_count += int(use_attention)
        fusion_dim = channels * branch_count
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def _encode(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        feat = encoder.forward_features(x)
        return self.pool(feat).flatten(1)

    def forward(
        self,
        image: torch.Tensor,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        mouth: torch.Tensor = None,
    ) -> torch.Tensor:
        global_map = self.global_encoder.forward_features(image)
        global_feat = self.pool(global_map).flatten(1)
        features = [global_feat]
        if self.use_left_eye:
            features.append(self._encode(self.left_encoder, left_eye))
        if self.use_right_eye:
            features.append(self._encode(self.right_encoder, right_eye))
        if self.use_mouth:
            if mouth is None:
                mouth = torch.zeros_like(left_eye)
            features.append(self._encode(self.mouth_encoder, mouth))
        if self.use_attention:
            features.append(self.pool(self.rinet_attention(global_map)).flatten(1))
        return self.regressor(torch.cat(features, dim=1)).squeeze(1)
