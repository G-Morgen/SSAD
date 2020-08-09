import streamlit as st


class BoardWidget:
    def init_fn(self):
        fn_list = [fn for fn in dir(self) if "show" in fn]
        fn = st.sidebar.radio("Select Function", fn_list)
        return getattr(self, fn)
