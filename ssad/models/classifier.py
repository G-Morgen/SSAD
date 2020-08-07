import torch.nn as nn
import ssad.typehint as T


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1 * 256 * 256, 4096),  # TODO: inflexible
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )

    def forward(self, x: T.Tensor):

        return self.classifier(x)
