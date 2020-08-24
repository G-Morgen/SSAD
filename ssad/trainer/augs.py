import ssad.typehint as T
from ssad import albu


class TrainerAugs:

    cfg: T.Path

    def init_augs(self, data_type: str) -> T.Compose:

        augs = albu.load(self.cfg.augs[data_type].yaml, data_format="yaml")
        augs = self.update_augs(data_type, augs)
        self.save_augs(data_type, augs)
        return augs

    def update_augs(self, data_type: str, augs: T.Compose) -> T.Compose:

        for aug_name, args in self.cfg.augs[data_type].args.items():
            for i, aug in enumerate(augs):
                if aug.__class__.__name__ == aug_name:
                    for k, v in args.items():
                        setattr(augs[i], k, v)
        return augs

    def save_augs(self, data_type: str, augs: T.Compose):

        albu.save(augs, f"hydra/{data_type}_augs.yaml", data_format="yaml")
