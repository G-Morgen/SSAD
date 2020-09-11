import torch
from tqdm import tqdm

import ssad.typehint as T


class TrainerRunTest:

    model: T.Module
    dataloader: T.DataLoader
    cfg: T.DictConfig
    log: T.Logger

    def run_test(self) -> None:

        self.model["S"].eval()
        self.model["C"].eval()
        pbar = tqdm(self.dataloader["test"], desc="test")
        for idx, data in enumerate(pbar):
            with torch.no_grad():
                img = data["image"].to(self.cfg.device)
                mask = data["mask"].to(self.cfg.device)
                label = data["label"].item()
                stem = data["stem"][0]

                semseg = self.model["S"](img)
                pred = self.model["C"](semseg)

                semseg = torch.argmax(semseg, 1)
                pred = torch.argmax(pred).item()

                self.show_result(stem, img, mask, label, semseg, pred)

                IoU = self.compute_IoU(semseg, mask)
                self.log.info(f"{stem} - {label} - {pred} - {IoU}")
