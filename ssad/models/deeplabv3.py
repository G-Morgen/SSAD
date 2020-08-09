import torch
import torch.nn as nn

from torchvision import models


class DeepLabV3(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)

        w = nn.Parameter(torch.empty(num_classes, 256, 3, 3))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        b = nn.Parameter(torch.empty(num_classes))
        b = nn.init.constant_(b, 0)
        self.model.classifier[-1].out_channels = num_classes
        self.model.classifier[-1].weight = w
        self.model.classifier[-1].bias = b

        w = nn.Parameter(torch.empty(num_classes, 256, 3, 3))
        w = nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        b = nn.Parameter(torch.empty(num_classes))
        b = nn.init.constant_(b, 0)
        self.model.aux_classifier[-1].out_channels = num_classes
        self.model.aux_classifier[-1].weight = w
        self.model.aux_classifier[-1].bias = b

    def forward(self, x):
        return self.model(x)["out"]
