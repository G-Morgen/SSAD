import albumentations as albu
import torch
import matplotlib.pyplot as plt
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

        self.augs = {}
        self.augs["train"] = self.get_augs(train_or_test="train")
        self.augs["test"] = self.get_augs(train_or_test="test")

        self.dataloader = {}
        self.dataloader["train"] = self.get_dataloader(train_or_test="train")
        self.dataloader["test"] = self.get_dataloader(train_or_test="test")
        self.model = self.get_model()
        self.model = self.model.to(self.cfg.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

    def get_model(self) -> T.Module:

        return deeplabv3.models.DeepLabV3()

    def get_augs(self, train_or_test: str) -> T.Compose:

        return albu.load(self.cfg[train_or_test]["augs"], data_format="yaml")

    def get_dataloader(self, train_or_test: str) -> T.DataLoader:

        dataset = SomicDataset(
            cfg=self.cfg, train_or_test=train_or_test, augs=self.augs[train_or_test]
        )
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
        pbar = tqdm(range(self.cfg.train.epochs), desc="train")
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
        for idx, sample in enumerate(pbar):
            with torch.no_grad():
                img = sample["image"].to(self.cfg.device)
                mask = sample["mask"].to(self.cfg.device)
                pred = self.model(img)["out"]
                self.show_test_result(idx, img, mask, pred)

    def show_test_result(self, idx: int, img: T.Tensor, mask: T.Tensor, pred: T.Tensor) -> None:

        img = self.unnormalize(img.squeeze())
        img = img.permute(1, 2, 0).cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(img)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(132)
        plt.imshow(img)
        plt.imshow(mask, cmap="Reds", alpha=0.5)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(133)
        plt.imshow(img)
        plt.imshow(pred, cmap="jet", alpha=0.5)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.tight_layout()
        plt.savefig(f"{idx}_test_result.png")

        print(img.shape)
        print(mask.shape)
        print(pred.shape)

    def unnormalize(self, tensor: T.Tensor) -> T.Tensor:

        for aug in self.augs["test"]:
            if aug.__class__.__name__ == "Normalize":
                mean = aug.mean
                std = aug.std

        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

        return tensor
