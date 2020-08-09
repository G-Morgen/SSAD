import torch

import ssad.models


class TrainerModel:
    def init_model(self, model_type: str):

        model_class = getattr(ssad.models, self.cfg.model[model_type].name)
        model = model_class(**self.cfg.model[model_type].args)
        return model.to(self.cfg.device)

    def load_pretrained_model(self, model, cfg):

        return model.load_state_dict(torch.load(cfg.pth))
