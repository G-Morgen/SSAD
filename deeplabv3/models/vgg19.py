import torch
import torch.nn as nn

from torchvision import models


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(pretrained=True)

        w = nn.Parameter(torch.empty(64, 2, 3, 3))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        self.model.features[0].in_channels = 2
        self.model.features[0].weight = w

        w = nn.Parameter(torch.empty(2, 4096))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        self.model.classifier[-1].out_features = 2
        self.model.classifier[-1].weight = w

        b = nn.Parameter(torch.empty(2))
        b = nn.init.constant_(b, 0)
        self.model.classifier[-1].bias = b

    def forward(self, x):
        return self.model(x)
