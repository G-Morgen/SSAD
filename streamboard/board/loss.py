import ssad.typehint as T
import streamlit as st
import pandas as pd


class BoardLoss:

    base: T.Path
    S_loss: T.DataFrame
    C_loss: T.DataFrame

    @st.cache(suppress_st_warning=True)
    def init_loss(self) -> T.List[T.DataFrame]:

        with open(self.base / "run.log", "r") as f:
            lines = f.readlines()

        S_loss_di: dict = {"epoch": [], "loss": []}
        C_loss_di: dict = {"epoch": [], "loss": []}
        for line in lines:
            line = line.replace("\n", "")
            func_name = line.split(" - ")[1]

            if func_name == "run_train_semseg":
                _, _, epoch, loss = line.split(" - ")
                S_loss_di["epoch"].append(int(epoch))
                S_loss_di["loss"].append(float(loss))

            if func_name == "run_train_classifier":
                _, _, epoch, loss = line.split(" - ")
                C_loss_di["epoch"].append(int(epoch))
                C_loss_di["loss"].append(float(loss))

        S_loss_df = pd.DataFrame(S_loss_di)
        S_loss_df = S_loss_df.groupby("epoch")
        S_loss_df = S_loss_df.mean()

        C_loss_df = pd.DataFrame(C_loss_di)
        C_loss_df = C_loss_df.groupby("epoch")
        C_loss_df = C_loss_df.mean()

        return [S_loss_df, C_loss_df]

    def show_loss(self):

        st.header("SemSeg Loss")
        st.line_chart(self.S_loss)

        st.header("Classifier Loss")
        st.line_chart(self.C_loss)
