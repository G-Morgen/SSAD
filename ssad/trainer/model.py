import torch

import ssad.models
import ssad.typehint as T


class TrainerModel:

    cfg: T.DictConfig
    model: T.Module

    def init_model(self, model_type: str):

        model_class = getattr(ssad.models, self.cfg.model[model_type].name)
        model = model_class(**self.cfg.model[model_type].args)
        return model.to(self.cfg.device)

    def load_pretrained_model(self):

        self.model["S"].load_state_dict(torch.load(self.cfg.model.S.pth))
        self.model["C"].load_state_dict(torch.load(self.cfg.model.C.pth))
