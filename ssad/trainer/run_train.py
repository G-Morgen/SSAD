class RunTrain:
    def run_train(self) -> None:

        pbar = tqdm(range(self.cfg.run_train.epoch), desc="train")
        for epoch in pbar:

            # Train semseg model
            self.model["S"].train()
            self.model["C"].train()
            for sample in self.dataloader["S"]:
                img = sample["image"].to(self.cfg.device)
                mask = sample["mask"].float().to(self.cfg.device)
                semseg = self.model["S"](img)
                loss = self.criterion["S"](semseg, mask)
                loss.backward()
                self.optimizer["S"].step()
                self.optimizer["S"].zero_grad()

            # Train classifier
            self.model["S"].eval()
            self.model["C"].train()
            for sample in self.dataloader["C"]:
                img = sample["image"].to(self.cfg.device)
                label = sample["label"].to(self.cfg.device)

                with torch.no_grad():
                    semseg = self.model["S"](img)

                pred = self.model["C"](semseg.view(semseg.shape[0], -1))
                loss = self.criterion["C"](pred, label)
                loss.backward()
                self.optimizer["C"].step()
                self.optimizer["C"].zero_grad()

        torch.save(self.model["S"].state_dict(), "semseg.pth")
        torch.save(self.model["C"].state_dict(), "classifier.pth")

