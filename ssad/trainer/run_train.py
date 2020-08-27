import mlflow
import torch
from tqdm import tqdm


class TrainerRunTrain:
    def run_train(self) -> None:

        pbar = tqdm(range(self.cfg.run_train.epoch), desc="train")
        for epoch in pbar:
            self.run_train_semseg(epoch)
            self.run_train_classifier(epoch)

        torch.save(self.model["S"].state_dict(), "semseg.pth")
        torch.save(self.model["C"].state_dict(), "classifier.pth")

    def run_train_semseg(self, epoch: int) -> None:

        self.model["S"].train()
        self.model["C"].train()
        for sample in self.dataloader["S"]:
            img = sample["image"].to(self.cfg.device)
            mask = sample["mask"].to(self.cfg.device)
            semseg = self.model["S"](img)
            loss = self.criterion["S"](semseg, mask)
            loss.backward()
            self.optimizer["S"].step()
            self.optimizer["S"].zero_grad()
            self.log.info(f"{epoch} - {loss}")
            mlflow.log_metric("loss", loss.detach().item())

    def run_train_classifier(self, epoch: int) -> None:

        self.model["S"].eval()
        self.model["C"].train()
        for sample in self.dataloader["C"]:
            img = sample["image"].to(self.cfg.device)
            label = sample["label"].to(self.cfg.device)

            with torch.set_grad_enabled(self.cfg.run_train.grad_enabled):
                semseg = self.model["S"](img)

            pred = self.model["C"](semseg)
            loss = self.criterion["C"](pred, label)
            loss.backward()
            self.optimizer["C"].step()
            self.optimizer["C"].zero_grad()
            self.log.info(f"{epoch} - {loss}")
