import os
import sys

import hydra

import ssad.typehint as T
from ssad.trainer import Trainer


config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def my_app(cfg: T.DictConfig) -> None:

    os.rename(".hydra", "hydra")
    trainer = Trainer(cfg)

    if cfg.model.S.pth and cfg.model.C.pth:
        trainer.load_model_pth()
    else:
        trainer.run_train()

    trainer.run_test()


if __name__ == "__main__":
    my_app()
