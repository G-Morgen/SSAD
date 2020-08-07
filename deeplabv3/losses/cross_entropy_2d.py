import torch.nn as nn
import torch.nn.functional as F
import deeplabv3.typehint as T


class CrossEntropy2D(nn.Module):
    def __init__(self, reduction: str = "mean"):

        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: T.Tensor, target: T.Tensor) -> T.Tensor:

        n, c, h, w = prediction.size()
        prediction = prediction.permute(0, 2, 3, 1).contiguous()
        prediction = prediction.view(-1, c)
        target = target.view(-1)
        target[target != 0] = 1
        loss = F.cross_entropy(prediction, target, reduction=self.reduction)
        return loss
