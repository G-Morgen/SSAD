import mlflow
import pandas as pd

import ssad.typehint as T


class TrainerConfusionMatrix:
    def compute_confusion_matrix(self):
        df = self._load_log()
        self._compute_confusion_matrix_for_image_classification(df)
        self._compute_confusion_matrix_for_product_classification(df)

    def _compute_confusion_matrix_for_image_classification(self, df: T.DataFrame):

        df["confusion_matrix"] = -1
        df.loc[(df["label"] == 1) & (df["pred"] == 1), "confusion_matrix"] = "TP"
        df.loc[(df["label"] == 0) & (df["pred"] == 1), "confusion_matrix"] = "FP"
        df.loc[(df["label"] == 1) & (df["pred"] == 0), "confusion_matrix"] = "FN"
        df.loc[(df["label"] == 0) & (df["pred"] == 0), "confusion_matrix"] = "TN"

        confusion_matrix = df["cm_image"].tolist()
        TP = confusion_matrix.count("TP")
        FP = confusion_matrix.count("FP")
        FN = confusion_matrix.count("FN")
        TN = confusion_matrix.count("TN")

        accuraccy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)

        mlflow.log_metric("TP_image", TP)
        mlflow.log_metric("FP_image", FP)
        mlflow.log_metric("FN_image", FN)
        mlflow.log_metric("TN_image", TN)

        mlflow.log_metric("accuracy_image", accuraccy)
        mlflow.log_metric("recall_image", recall)
        mlflow.log_metric("specificity_image", specificity)
        mlflow.log_metric("precision_image", precision)

    def _compute_confusion_matrix_for_product_classification(self, df: T.DataFrame):

        df["confusion_matrix"] = -1
        df.loc[(df["label_sum"] != 0) & (df["pred_sum"] != 0), "confusion_matrix"] = "TP"
        df.loc[(df["label_sum"] == 0) & (df["pred_sum"] != 0), "confusion_matrix"] = "FP"
        df.loc[(df["label_sum"] != 0) & (df["pred_sum"] == 0), "confusion_matrix"] = "FN"
        df.loc[(df["label_sum"] == 0) & (df["pred_sum"] == 0), "confusion_matrix"] = "TN"

        confusion_matrix = df.loc[df["angle"] == 0, "confusion_matrix"].tolist()
        TP = confusion_matrix.count("TP")
        FP = confusion_matrix.count("FP")
        FN = confusion_matrix.count("FN")
        TN = confusion_matrix.count("TN")

        accuraccy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)

        mlflow.log_metric("TP_product", TP)
        mlflow.log_metric("FP_product", FP)
        mlflow.log_metric("FN_product", FN)
        mlflow.log_metric("TN_product", TN)

        mlflow.log_metric("accuracy_product", accuraccy)
        mlflow.log_metric("recall_product", recall)
        mlflow.log_metric("specificity_product", specificity)
        mlflow.log_metric("precision_product", precision)

    def _load_log(self) -> T.DataFrame:

        with open("run.log", "r") as f:
            lines = f.readlines()

        di: dict = {"stem": [], "label": [], "pred": [], "IoU": []}
        for line in lines:
            line = line.replace("\n", "")

            if len(line.split(" - ")) == 5:
                _, stem, label, pred, IoU = line.split(" - ")
                di["stem"].append(stem)
                di["label"].append(int(label))
                di["pred"].append(int(pred))
                di["IoU"].append(float(IoU))

        df = pd.DataFrame(di)
        df["product_id"] = df["stem"].apply(lambda x: int(x.split("_")[0]))
        df["crop_type"] = df["stem"].apply(lambda x: int(x.split("_")[1]))
        df["angle"] = df["stem"].apply(lambda x: int(x.split("_")[2]))

        df["label_sum"] = -1
        df["pred_sum"] = -1
        for product_id in df["product_id"].unique():
            expr = df["product_id"] == product_id
            df.loc[expr, "label_sum"] = df.loc[expr, "label"].sum()
            df.loc[expr, "pred_sum"] = df.loc[expr, "pred"].sum()

        return df
