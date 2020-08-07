import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


def show_test_anomaly_result(base: Path):
    @st.cache(suppress_st_warning=True)
    def get_product_list(base):
        product_list = []
        for p in base.glob("*_anomaly_*.png"):
            _, _, _, _, idx, _ = p.stem.split("_")
            product_list.append(int(idx))
        return product_list

    @st.cache(suppress_st_warning=True)
    def get_angle_minmax(base):
        angle_list = []
        for p in base.glob("*_anomaly_*.png"):
            _, _, _, _, _, angle = p.stem.split("_")
            angle_list.append(int(angle))
        return angle_list

    product_list = get_product_list(base)
    i = st.slider(label="product", min_value=0, max_value=len(product_list), value=0, step=1)
    product = product_list[i]

    angle_list = get_angle_minmax(base)
    angle = st.slider(
        label="angle",
        min_value=min(angle_list),
        max_value=max(angle_list),
        value=min(angle_list),
        step=1,
    )

    for p in base.glob(f"*_anomaly_{product}_{angle}.png"):
        st.write(p.stem)
        result = Image.open(p)

    plt.figure(figsize=(9, 3))
    plt.imshow(result)
    plt.axis("off")
    st.pyplot()
