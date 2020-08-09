import streamlit as st
import pandas as pd
import ssad.typehint as T


class BoardConfusionMatrix:

    base: T.Path
    classifier_result: T.DataFrame

    @st.cache(suppress_st_warning=True)
    def init_confusion_matrix(self) -> T.DataFrame:

        with open(self.base / "run.log", "r") as f:
            lines = f.readlines()

        di: dict = {"stem": [], "label": [], "pred": []}
        for line in lines:
            line = line.replace("\n", "")
            func_name = line.split(" - ")[1]

            if func_name == "run_test":
                _, _, stem, label, pred, _ = line.split(" - ")
                di["stem"].append(stem)
                di["label"].append(int(label))
                di["pred"].append(int(pred))

        df = pd.DataFrame(di)
        df["angle"] = df["stem"].apply(lambda x: int(x.split("_")[-1]))
        df["product"] = df["stem"].apply(lambda x: int(x.split("_")[-2]))
        df["stem_"] = df["stem"].apply(lambda x: "_".join(x.split("_")[:-1]))

        df["label_sum"] = -1
        df["pred_sum"] = -1
        for stem_ in df["stem_"].unique():
            tmp = df[df["stem_"] == stem_]
            df.loc[df["stem_"] == stem_, "label_sum"] = tmp["label"].sum()
            df.loc[df["stem_"] == stem_, "pred_sum"] = tmp["pred"].sum()

        df["cm_for_image"] = -1
        df.loc[(df["label"] == 0) & (df["pred"] == 0), "cm_for_image"] = "TN"
        df.loc[(df["label"] == 0) & (df["pred"] == 1), "cm_for_image"] = "FP"
        df.loc[(df["label"] == 1) & (df["pred"] == 0), "cm_for_image"] = "FN"
        df.loc[(df["label"] == 1) & (df["pred"] == 1), "cm_for_image"] = "TP"

        df["cm_for_product"] = -1
        df.loc[(df["label_sum"] == 0) & (df["pred_sum"] == 0), "cm_for_product"] = "TN"
        df.loc[(df["label_sum"] == 0) & (df["pred_sum"] != 0), "cm_for_product"] = "FP"
        df.loc[(df["label_sum"] != 0) & (df["pred_sum"] == 0), "cm_for_product"] = "FN"
        df.loc[(df["label_sum"] != 0) & (df["pred_sum"] != 0), "cm_for_product"] = "TP"

        return df.sort_values(by=["product", "angle"])

    def show_confusion_matrix(self):

        cm = pd.DataFrame({}, index=["TP", "FP", "FN", "TN"])
        cm["cm_for_image"] = self.classifier_result["cm_for_image"].value_counts()
        cm["cm_for_product"] = self.classifier_result["cm_for_product"].value_counts() // 12
        cm = cm.fillna(0).T
        cm["accuracy"] = (cm.TP + cm.TN) / (cm.TP + cm.FP + cm.FN + cm.TN)
        cm["recall"] = cm.TP / (cm.TP + cm.FN)
        cm["specificity"] = cm.TN / (cm.TN + cm.FP)
        cm["precision"] = cm.TP / (cm.TP + cm.FP)
        st.dataframe(cm)
