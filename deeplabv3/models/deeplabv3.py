import torch
import torch.nn as nn

from torchvision import models


class DeepLabV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)

        w = nn.Parameter(torch.empty(2, 256, 3, 3))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        self.model.aux_classifier[-1].weight = w

        b = nn.Parameter(torch.empty(2))
        b = nn.init.constant_(b, 0)
        self.model.aux_classifier[-1].bias = b

        w = nn.Parameter(torch.empty(2, 256, 3, 3))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        self.model.aux_classifier[-1].weight = w

        b = nn.Parameter(torch.empty(2))
        b = nn.init.constant_(b, 0)
        self.model.classifier[-1].bias = b

        w = nn.Parameter(torch.empty(2, 256, 3, 3))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        self.model.classifier[-1].weight = w

    def forward(self, x):
        return self.model(x)["out"]
