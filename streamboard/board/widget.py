import streamlit as st
from pathlib import Path


class BoardWidget:
    def init_base(self) -> Path:

        base = Path("/dgx/github/SSAD/ssad/experiment")
        experiment_list = [p for p in base.glob("*/outputs/*/*")]
        experiment = st.sidebar.selectbox("Select Experiment", experiment_list)
        return experiment

    def init_fn(self):
        fn_list = [fn for fn in dir(self) if "show" in fn]
        fn = st.sidebar.selectbox("Select Function", fn_list)
        return getattr(self, fn)
