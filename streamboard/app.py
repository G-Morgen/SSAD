import streamlit as st
import streamboard.utils

from pathlib import Path


st.title("DeepLabV3")
base = Path("/dgx/github/DeepLabV3/deeplabv3/outputs/2020-08-07/11-51-17")

func_name = st.sidebar.selectbox(
    "Function", ("show_test_anomaly_result", "show_test_normal_result", "show_metrics")
)

st.header(" ")
st.header(" ")
fn = getattr(streamboard.utils, func_name)
fn(base)
