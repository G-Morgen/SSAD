from sklearn.metrics import jaccard_score

import ssad.typehint as T


class TrainerMetric:
    def compute_IoU(self, semseg: T.Tensor, mask: T.Tensor) -> float:

        semseg = semseg.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        mask[mask != 0] = 1
        return jaccard_score(mask.flatten(), semseg.flatten(), average="binary")
