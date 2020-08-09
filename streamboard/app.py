import streamlit as st

from pathlib import Path
from streamboard.board import Board

st.title("SemSeg for Anomaly Detection")
st.header(" ")

base = Path("/dgx/github/SSAD/ssad/outputs/2020-08-09/15-32-13")
board = Board(base)
board.fn()
