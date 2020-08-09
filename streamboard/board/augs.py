import streamlit as st
import matplotlib.pyplot as plt
import ssad.typehint as T
from omegaconf import OmegaConf
from ssad import albu


class BoardAugs:

    base: T.Path
    cfg: T.DictConfig

    def init_augs(self, data_type: str):

        return albu.load(self.base / f"hydra/{data_type}_augs.yaml", data_format="yaml")

    def show_augs(self):

        data_type = st.selectbox("Data Type", ["C", "S", "test"])
        dataset = self.dataset[data_type]
        idx = st.slider(label="Data ID", min_value=0, max_value=len(dataset), value=0, step=1)
        stem = dataset[idx]["stem"]
        img = dataset[idx]["image"].permute(1, 2, 0)
        mask = dataset[idx]["mask"].squeeze()

        plt.figure(figsize=(9, 3))
        plt.subplot(131)
        plt.imshow(img)
        plt.title("Image")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(132)
        plt.imshow(mask, cmap="Reds")
        plt.title("Mask")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(133)
        plt.imshow(img)
        plt.imshow(mask, cmap="Reds", alpha=0.3)
        plt.title("Supervision")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        st.write(stem)
        st.pyplot()

        augs_yaml = OmegaConf.load(str(self.base / f"hydra/{data_type}_augs.yaml"))
        st.code(augs_yaml.pretty(), language="yaml")
