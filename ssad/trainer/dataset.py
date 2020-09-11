import ssad.typehint as T
from ssad.dataset import SomicDataset


class TrainerDataset:

    cfg: T.DictConfig
    augs: T.Compose

    def init_dataset(self, data_type: str) -> T.Dataset:

        dataset = SomicDataset(self.cfg.dataset[data_type], self.augs[data_type])
        return dataset
