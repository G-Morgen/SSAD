import streamlit as st

from pathlib import Path
from streamboard.board import Board

st.title("SemSeg for Anomaly Detection")
st.header(" ")

base = Path("/dgx/github/SSAD/ssad/experiment/resolutions/outputs/2020-08-10/21-18-35")
board = Board(base)
board.fn()
