import matplotlib.pyplot as plt
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

import ssad.typehint as T
import ssad.models
import ssad.losses
from deeplabv3 import albu
from deeplabv3.dataset import SomicDataset


class Trainer:
    def __init__(self, cfg: T.DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.augs = {}
        self.dataset = {}
        self.dataloader = {}
        for data_type in ["S", "C", "test"]:
            self.augs[data_type] = self.get_augs(data_type)
            self.dataset[data_type] = self.get_dataset(data_type)
            self.dataloader[data_type] = self.get_dataloader(data_type)

        self.model = {}
        self.optimizer = {}
        self.criterion = {}
        for data_type in ["S", "C"]:
            self.model[data_type] = self.get_model(data_type)
            self.optimizer[data_type] = self.get_optimizer(data_type)
            self.criterion[data_type] = self.get_criterion(data_type)

    def get_augs(self, data_type: str) -> T.Compose:

        return albu.load(self.cfg.augs[data_type], data_format="yaml")

    def get_dataset(self, data_type: str):

        return SomicDataset(self.cfg.dataset[data_type], self.augs[data_type])

    def get_dataloader(self, data_type: str) -> T.DataLoader:

        dataloader = DataLoader(
            dataset=self.dataset[data_type],
            batch_size=self.cfg.dataloader[data_type].batch_size,
            shuffle=self.cfg.dataloader[data_type].shuffle,
        )
        return dataloader

    def get_model(self, data_type: str):

        model = getattr(deeplabv3.models, self.cfg.model[data_type].name)
        return model().to(self.cfg.device)

    def get_optimizer(self, data_type: str) -> T.Optimizer:

        params = self.model[data_type].parameters()
        optimizer = getattr(torch.optim, self.cfg.optimizer[data_type].name)
        args = self.cfg.optimizer[data_type].args
        if args:
            return optimizer(params, **args)
        else:
            return optimizer(params)

    def get_criterion(self, data_type: str):

        criterion = getattr(deeplabv3.losses, self.cfg.criterion[data_type].name)
        args = self.cfg.criterion[data_type].args
        if args:
            return criterion(**args)
        else:
            return criterion()

    def load_model_pth(self) -> None:

        self.model["S"].load_state_dict(torch.load(self.cfg.model.S.pth))
        self.model["C"].load_state_dict(torch.load(self.cfg.model.C.pth))

    def run_train(self) -> None:

        pbar = tqdm(range(self.cfg.run_train.epoch), desc="train")
        for epoch in pbar:

            # Train semseg model
            self.model["S"].train()
            self.model["C"].train()
            for data in self.dataloader["S"]:
                img = data["image"].to(self.cfg.device)
                mask = data["mask"].long().to(self.cfg.device)
                semseg = self.model["S"](img)
                loss = self.criterion["S"](semseg, mask)
                loss.backward()
                self.optimizer["S"].step()
                self.optimizer["S"].zero_grad()

            # Train classifier
            self.model["S"].eval()
            self.model["C"].train()
            for data in self.dataloader["C"]:
                img = data["image"].to(self.cfg.device)
                label = data["label"].to(self.cfg.device)

                with torch.no_grad():
                    semseg = self.model["S"](img)

                pred = self.model["C"](semseg)
                loss = self.criterion["C"](pred, label)
                loss.backward()
                self.optimizer["C"].step()
                self.optimizer["C"].zero_grad()

        torch.save(self.model["S"].state_dict(), "semseg.pth")
        torch.save(self.model["C"].state_dict(), "classifier.pth")

    def run_test(self) -> None:

        self.model["S"].eval()
        self.model["C"].eval()
        di: dict = {"label": [], "pred": [], "stem": []}
        pbar = tqdm(self.dataloader["test"], desc="test")
        for idx, data in enumerate(pbar):
            with torch.no_grad():
                img = data["image"].to(self.cfg.device)
                mask = data["mask"].to(self.cfg.device)
                label = data["label"].item()
                stem = data["stem"][0]

                semseg = self.model["S"](img)
                pred = self.model["C"](semseg)

                semseg = torch.argmax(semseg, 1)
                pred = torch.argmax(pred).item()

                di["label"].append(label)
                di["pred"].append(pred)
                di["stem"].append(stem)

                self.show_test_result(stem, img, mask, label, semseg, pred)

        df = pd.DataFrame(di)
        df.to_csv("result.csv")

    def show_test_result(
        self, stem: str, img: T.Tensor, mask: T.Tensor, label: int, semseg: T.Tensor, pred: int,
    ) -> None:

        img = self.unnormalize(img.squeeze())
        img = img.permute(1, 2, 0).cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        semseg = semseg.squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.imshow(img)
        plt.title("Image")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(132)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.3, cmap="Reds")
        plt.title(f"Mask (Label={label})")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(133)
        plt.imshow(img)
        plt.imshow(semseg, alpha=0.3, cmap="Reds")
        plt.title(f"Semseg (Pred={pred})")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.savefig(f"{stem}.png", bbox_inches="tight")
        plt.close()

    def unnormalize(self, tensor: T.Tensor) -> T.Tensor:

        for aug in self.augs["test"]:
            if aug.__class__.__name__ == "Normalize":
                mean = aug.mean
                std = aug.std

        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

        return tensor
