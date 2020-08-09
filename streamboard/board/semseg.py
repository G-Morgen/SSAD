import streamlit as st
import matplotlib.pyplot as plt
import ssad.typehint as T
from PIL import Image


class BoardSemseg:

    base: T.Path
    classifier_result: T.DataFrame

    def show_semseg(self):

        result_type = st.selectbox(" ", ["TP", "FP", "FN", "TN"])
        df = self.classifier_result.query("cm_for_product==@result_type")
        product_list = df["product"].unique()
        i = st.slider(
            label="product", min_value=0, max_value=len(product_list) - 1, value=0, step=1,
        )
        angle = st.slider(
            label="angle", min_value=0, max_value=len(df["angle"].unique()) - 1, value=0, step=1,
        )
        stem = df.query(f"product=={product_list[i]} & angle=={angle}")["stem"]
        img_path = self.base / f"{stem.item()}.png"
        st.write(stem.item())
        result = Image.open(img_path)
        plt.figure(figsize=(9, 3))
        plt.imshow(result)
        plt.axis("off")
        plt.tight_layout()
        st.pyplot()
