import albumentations as albu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import deeplabv3.models
import deeplabv3.typehint as T
from deeplabv3 import albu
from deeplabv3.dataset import SomicDataset


class Trainer:
    def __init__(self, cfg: T.DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.dataloader = {}
        self.dataloader["train"] = self.get_dataloader(train_or_test="train")
        self.dataloader["test"] = self.get_dataloader(train_or_test="test")
        self.model = self.get_model()
        self.model = self.model.to(self.cfg.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

    def get_model(self) -> T.Module:

        return deeplabv3.models.DeepLabV3()

    def get_dataloader(self, train_or_test: str) -> T.DataLoader:

        augs = albu.load(self.cfg[train_or_test]["augs"], data_format="yaml")
        dataset = SomicDataset(cfg=self.cfg, train_or_test=train_or_test, augs=augs)
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.cfg[train_or_test].batch_size, shuffle=True
        )
        return dataloader

    def get_optimizer(self) -> T.Optimizer:

        parameters = self.model.parameters()
        lr = self.cfg.train.optim.lr
        weight_decay = self.cfg.train.optim.weight_decay
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_criterion(self) -> T.Loss:

        return torch.nn.MSELoss(reduction="mean")

    def run_train(self) -> None:

        self.model.train()
        pbar = tqdm(range(1, self.cfg.train.epochs), desc="train")
        for epoch in pbar:
            for sample in self.dataloader["train"]:
                img = sample["image"].to(self.cfg.device)
                mask = sample["mask"].float().to(self.cfg.device)
                pred = self.model(img)["out"]
                loss = self.criterion(pred, mask)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def run_test(self) -> None:

        self.model.eval()
        pbar = tqdm(self.dataloader["test"], desc="test")
        for sample in pbar:
            img = sample["image"].to(self.cfg.device)
            mask = sample["mask"].to(self.cfg.device)
            pred = self.model(img)
