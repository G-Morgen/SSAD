import os
import sys
from pathlib import Path

import hydra
import mlflow

import ssad.typehint as T
from ssad.trainer import Trainer

config_path = sys.argv[1]
sys.argv.pop(1)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/taikiinoue45@gmail.com/my-experiment")


@hydra.main(config_path)
def my_app(cfg: T.DictConfig) -> None:

    os.rename(".hydra", "hydra")
    trainer = Trainer(cfg)

    if cfg.model.S.pth and cfg.model.C.pth:
        trainer.load_pretrained_model()
    else:
        trainer.run_train()

    trainer.run_test()

    mlflow.log_artifact(Path(".").parent)


if __name__ == "__main__":
    my_app()
