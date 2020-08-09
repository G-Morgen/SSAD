import matplotlib.pyplot as plt

import ssad.typehint as T


class TrainerShowResult:
    def show_result(
        self, stem: str, img: T.Tensor, mask: T.Tensor, label: int, semseg: T.Tensor, pred: int
    ) -> None:

        img = self.unnormalize(img.squeeze())
        img = img.permute(1, 2, 0).cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        semseg = semseg.squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.imshow(img)
        plt.title("Image")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(132)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.3, cmap="Reds")
        plt.title(f"Mask (Label={label})")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(133)
        plt.imshow(img)
        plt.imshow(semseg, alpha=0.3, cmap="Reds")
        plt.title(f"Semseg (Pred={pred})")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.savefig(f"{stem}.png", bbox_inches="tight")
        plt.close()

    def unnormalize(self, tensor: T.Tensor) -> T.Tensor:

        for aug in self.augs["test"]:
            if aug.__class__.__name__ == "Normalize":
                mean = aug.mean
                std = aug.std

        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

        return tensor
