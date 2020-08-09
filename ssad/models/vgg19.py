import torch
import torch.nn as nn

from torchvision import models


class VGG19(nn.Module):
    def __init__(self, pretrained: bool, in_channels: int, out_features: int):
        super().__init__()
        self.model = models.vgg19(pretrained=pretrained)

        w = nn.Parameter(torch.empty(64, in_channels, 3, 3))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        self.model.features[0].in_channels = in_channels
        self.model.features[0].weight = w

        w = nn.Parameter(torch.empty(out_features, 4096))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        b = nn.Parameter(torch.empty(out_features))
        b = nn.init.constant_(b, 0)
        self.model.classifier[-1].out_features = out_features
        self.model.classifier[-1].weight = w
        self.model.classifier[-1].bias = b

    def forward(self, x):
        return self.model(x)
