import ssad.typehint as T
import streamlit as st
from omegaconf import OmegaConf


class BoardConfig:

    base: T.Path
    cfg: T.DictConfig

    @st.cache(suppress_st_warning=True)
    def init_config(self) -> T.DictConfig:

        return OmegaConf.load(str(self.base / "hydra/config.yaml"))

    def show_config(self):

        st.code(self.cfg.pretty(), language="yaml")
