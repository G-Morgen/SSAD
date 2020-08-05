import os
import sys

import hydra

import deeplabv3.typehint as T
from deeplabv3.trainer import Trainer


config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def my_app(cfg: T.DictConfig) -> None:

    os.rename(".hydra", "hydra")
    trainer = Trainer(cfg)
    trainer.run_train()
    trainer.run_test()


if __name__ == "__main__":
    my_app()
