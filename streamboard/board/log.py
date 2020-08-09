import streamlit as st
import pandas as pd


class BoardLoadLog:
    @st.cache(suppress_st_warning=True)
    def load_log(self):

        with open(self.base / "run.log", "r") as f:
            lines = f.readlines()

        S_losses: dict = {"epoch": [], "loss": []}
        C_losses: dict = {"epoch": [], "loss": []}
        C_results: dict = {"stem": [], "label": [], "pred": []}
        IoU_list = []
        for line in lines:
            line = line.replace("\n", "")
            func_name = line.split(" - ")[1]

            if func_name == "run_train_semseg":
                _, _, epoch, loss = line.split(" - ")
                S_losses["epoch"].append(int(epoch))
                S_losses["loss"].append(float(loss))

            if func_name == "run_train_classifier":
                _, _, epoch, loss = line.split(" - ")
                C_losses["epoch"].append(int(epoch))
                C_losses["loss"].append(float(loss))

            if func_name == "run_test":
                _, _, stem, label, pred, IoU = line.split(" - ")
                C_results["stem"].append(stem)
                C_results["label"].append(label)
                C_results["pred"].append(pred)
                if label == "1":
                    IoU_list.append(float(IoU))

        S_losses = pd.DataFrame(S_losses)
        S_losses = S_losses.groupby("epoch")
        S_losses = S_losses.mean()

        C_losses = pd.DataFrame(C_losses)
        C_losses = C_losses.groupby("epoch")
        C_losses = C_losses.mean()

        C_results = pd.DataFrame(C_results)
        C_results["confusion_matrix"] = -1
        C_results.query("label==0 & pred==0")["confusion_matrix"] = "TN"
        C_results.query("label==0 & pred==1")["confusion_matrix"] = "FP"
        C_results.query("label==1 & pred==0")["confusion_matrix"] = "FN"
        C_results.query("label==1 & pred==1")["confusion_matrix"] = "TP"

        mIoU = sum(IoU_list) / len(IoU_list)

        return S_losses, C_losses, C_results, mIoU
