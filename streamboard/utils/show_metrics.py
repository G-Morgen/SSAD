import streamlit as st
import pandas as pd
import numpy as np


def show_metrics(base):

    df = pd.read_csv(base / "result.csv")
    TP = len(df.loc[(df["label"] == 1) & (df["pred"] == 1)])
    FP = len(df.loc[(df["label"] == 0) & (df["pred"] == 1)])
    TN = len(df.loc[(df["label"] == 0) & (df["pred"] == 0)])
    FN = len(df.loc[(df["label"] == 1) & (df["pred"] == 0)])
    accuracy = round((TP + TN) / (TP + TN + FP + FN), 3)
    recall = round(TP / (TP + FN), 3)
    specificity = round(TN / (TN + FP), 3)
    precision = round(TP / (TP + FP), 3)

    df1 = pd.DataFrame(
        data={"Actual Anomaly": [TP, FN], "Actual Normal": [FP, TN]},
        index=["Classified Anomaly", "Classified Normal"],
    )

    df2 = pd.DataFrame(
        data={
            "Definition": ["(TP+TN)/(TP+TN+FP+FN)", "TP/(TP+FN)", "TN/(TN+FP)", "TP/(TP+FP)"],
            "Metrics": [accuracy, recall, specificity, precision],
        },
        index=["Accuracy", "Recall", "Specificity", "Precision"],
    )

    def func(df):
        df["pred_sum"] = -1
        df["label_sum"] = -1
        for product in df["product"].unique():
            df_ = df.loc[df["product"] == product]
            df.loc[df["product"] == product, "pred_sum"] = df_["pred"].sum()
            df.loc[df["product"] == product, "label_sum"] = df_["label"].sum()

        TP = len(df.loc[(df["label_sum"] != 0) & (df["pred_sum"] != 0)])
        FP = len(df.loc[(df["label_sum"] == 0) & (df["pred_sum"] != 0)])
        TN = len(df.loc[(df["label_sum"] == 0) & (df["pred_sum"] == 0)])
        FN = len(df.loc[(df["label_sum"] != 0) & (df["pred_sum"] == 0)])

        return np.array([TP, FP, TN, FN])

    df = pd.read_csv(base / "result.csv")
    df["angle"] = df["stem"].apply(lambda x: x.split("_")[-1])
    df["product"] = df["stem"].apply(lambda x: x.split("_")[-2])
    df["normal_or_anomaly"] = df["stem"].apply(lambda x: x.split("_")[-3])

    arr = np.zeros(4)
    df_normal = df.loc[df["normal_or_anomaly"] == "normal"]
    df_anomaly = df.loc[df["normal_or_anomaly"] == "anomaly"]
    arr += func(df_normal)
    arr += func(df_anomaly)
    arr /= 12
    TP, FP, TN, FN = arr

    accuracy = round((TP + TN) / (TP + TN + FP + FN), 3)
    recall = round(TP / (TP + FN), 3)
    specificity = round(TN / (TN + FP), 3)
    precision = round(TP / (TP + FP), 3)

    df3 = pd.DataFrame(
        data={"Actual Anomaly": [TP, FN], "Actual Normal": [FP, TN]},
        index=["Classified Anomaly", "Classified Normal"],
    )

    df4 = pd.DataFrame(
        data={
            "Definition": ["(TP+TN)/(TP+TN+FP+FN)", "TP/(TP+FN)", "TN/(TN+FP)", "TP/(TP+FP)"],
            "Metrics": [accuracy, recall, specificity, precision],
        },
        index=["Accuracy", "Recall", "Specificity", "Precision"],
    )

    st.header("Image-wise Confusion Matrix")
    st.dataframe(df1)
    st.header(" ")
    st.header("Image-wise Metrics")
    st.dataframe(df2)

    st.header(" ")
    st.header("Product-wise Confusion Matrix")
    st.dataframe(df3)
    st.header(" ")
    st.header("Product-wise Metrics")
    st.dataframe(df4)
