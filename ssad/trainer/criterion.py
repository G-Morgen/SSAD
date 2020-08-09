import ssad.typehint as T
import ssad.losses


class TrainerCriterion:
    def init_criterion(self, data_type: str) -> T.Loss:

        criterion = getattr(ssad.losses, self.cfg.criterion[data_type].name)
        args = self.cfg.criterion[data_type].args
        if args:
            return criterion(**args)
        else:
            return criterion()
