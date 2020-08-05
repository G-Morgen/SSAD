from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import deeplabv3.typehint as T


class SomicDataset(Dataset):
    def __init__(self, cfg: T.DictConfig, train_or_test: str, augs: albu.Compose) -> None:

        self.augs = augs
        self.base = Path(cfg.dataset.base)

        info = pd.read_csv(self.base / "info.csv")
        self.stems = []
        for di in cfg.dataset[train_or_test]:
            folder = di["name"]
            has_kizu = di["has_kizu"]

            stem = info.loc[(info["folder"] == folder) & (info["has_kizu"] == has_kizu), "stem"]
            self.stems += stem.to_list()

    def __getitem__(self, idx: int):

        stem = self.stems[idx]

        img = cv2.imread(str(self.base / f"images/{stem}.bmp"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.base / f"masks/{stem}.png"), cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        sample = self.augs(image=img, mask=mask)
        sample["mask"] = sample["mask"].permute(2, 0, 1)
        return sample

    def __len__(self) -> int:

        return len(self.stems)
