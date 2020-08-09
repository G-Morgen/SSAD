import streamlit as st
import pandas as pd
import ssad.typehint as T
from ssad.dataset import SomicDataset


class BoardDataset:

    cfg: T.DictConfig
    augs: T.Compose
    dataset: T.Dataset

    @st.cache(suppress_st_warning=True)
    def init_dataset(self, data_type: str) -> T.Dataset:

        return SomicDataset(self.cfg.dataset[data_type], self.augs[data_type])

    def show_dataset(self):

        data_type = st.selectbox("Data Type", ["C", "S", "test"])
        dataset = self.dataset[data_type]
        df = pd.DataFrame({data_type: dataset.stems})
        df["angle"] = df[data_type].apply(lambda x: x.split("_")[-1])
        df["product"] = df[data_type].apply(lambda x: x.split("_")[-2])
        df["is_normal"] = df[data_type].apply(lambda x: 1 if x.split("_")[-3] == "normal" else 0)
        st.dataframe(df)
