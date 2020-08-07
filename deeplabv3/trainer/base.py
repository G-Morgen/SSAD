import torch
from torch.utils.data import DataLoader

import deeplabv3.typehint as T
import deeplabv3.models
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

        criterion = getattr(torch.nn, self.cfg.criterion[data_type].name)
        args = self.cfg.criterion[data_type].args
        if args:
            return criterion(**args)
        else:
            return criterion()

    def load_model_pth(self) -> None:

        self.model["S"].load_state_dict(torch.load(self.cfg.model.S.pth))
        self.model["C"].load_state_dict(torch.load(self.cfg.model.C.pth))
