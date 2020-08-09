import torch

import ssad.typehint as T


class TrainerMetric:
    def compute_IoU(self, semseg: T.Tensor, mask: T.Tensor) -> float:

        eps = 1e-10
        intersection = torch.sum(semseg + mask == 2)
        union = torch.sum(semseg + mask != 0)
        iou = intersection / (union + eps)
        return iou.item()
