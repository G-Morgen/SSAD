import torch.nn as nn

from torchvision import models


class DeepLabV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)

    def forward(self, x):
        return self.model(x)
