import torch
import ssad.typehint as T


class TrainerOptimizer:
    def init_optimizer(self, model_type: str) -> T.Optimizer:

        params = self.model[model_type].parameters()
        optimizer = getattr(torch.optim, self.cfg.optimizer[model_type].name)
        args = self.cfg.optimizer[model_type].args
        if args:
            return optimizer(params, **args)
        else:
            return optimizer(params)
