from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import ssad.typehint as T


class SomicDataset(Dataset):
    def __init__(self, cfg: T.DictConfig, augs: T.Compose) -> None:

        self.augs = augs
        self.base = Path(cfg["base"])
        self.stems = []

        df = pd.read_csv(self.base / "info.csv")
        for query in cfg["query"]:
            stem = df.query(query)["stem"]
            self.stems += stem.to_list()

    def __getitem__(self, idx: int):

        stem = self.stems[idx]
        img = cv2.imread(str(self.base / f"images/{stem}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.base / f"masks/{stem}.png"), cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        label = 0 if mask.sum() == 0 else 1

        data = self.augs(image=img, mask=mask)
        data["mask"] = data["mask"].permute(2, 0, 1)
        data["label"] = label
        data["stem"] = stem
        return data

    def __len__(self) -> int:

        return len(self.stems)
