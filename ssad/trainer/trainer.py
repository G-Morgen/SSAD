import logging

import ssad.typehint as T
from ssad.trainer.augs import TrainerAugs
from ssad.trainer.dataset import TrainerDataset
from ssad.trainer.dataloader import TranerDataLoader
from ssad.trainer.model import TrainerModel
from ssad.trainer.optimizer import TrainerOptimizer
from ssad.trainer.criterion import TrainerCriterion
from ssad.trainer.run_train import TrainerRunTrain
from ssad.trainer.run_test import TrainerRunTest
from ssad.trainer.metric import TrainerMetric
from ssad.trainer.show_result import TrainerShowResult


class Trainer(
    TrainerAugs,
    TrainerDataset,
    TranerDataLoader,
    TrainerModel,
    TrainerOptimizer,
    TrainerCriterion,
    TrainerRunTrain,
    TrainerRunTest,
    TrainerMetric,
    TrainerShowResult,
):
    def __init__(self, cfg: T.DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.augs = {}
        self.dataset = {}
        self.dataloader = {}
        for data_type in ["S", "C", "test"]:
            self.augs[data_type] = self.init_augs(data_type)
            self.dataset[data_type] = self.init_dataset(data_type)
            self.dataloader[data_type] = self.init_dataloader(data_type)

        self.model = {}
        self.optimizer = {}
        self.criterion = {}
        for model_type in ["S", "C"]:
            self.model[model_type] = self.init_model(model_type)
            self.optimizer[model_type] = self.init_optimizer(model_type)
            self.criterion[model_type] = self.init_criterion(model_type)
