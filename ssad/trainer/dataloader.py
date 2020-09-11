from torch.utils.data import DataLoader

import ssad.typehint as T


class TranerDataLoader:

    dataset: T.Dataset
    cfg: T.DictConfig

    def init_dataloader(self, data_type: str) -> T.DataLoader:

        dataloader = DataLoader(
            dataset=self.dataset[data_type],
            batch_size=self.cfg.dataloader[data_type].batch_size,
            shuffle=self.cfg.dataloader[data_type].shuffle,
        )
        return dataloader
