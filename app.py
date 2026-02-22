import streamlit as st
import pandas as pd
import time

from data_handler import load_data, clean_data, scale_features
from models import detect_anomalies
from visualization import plot_pca, plot_scores
from explainability import explain_anomalies
from utils import filter_anomalies

st.set_page_config(page_title="Explainable Anomaly Detection", layout="wide")

st.title("Explainable Anomaly Detection Platform")

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:

    df = load_data(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df = clean_data(df)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    selected_cols = st.multiselect("Select Features", numeric_cols)

    model_name = st.selectbox(
        "Select Model",
        ["Isolation Forest"]
    )

    contamination = st.slider("Contamination", 0.01, 0.2, 0.05)

    if selected_cols:

        scaled_data = scale_features(df, selected_cols)

        start = time.time()
        labels, scores = detect_anomalies(model_name, scaled_data, contamination)
        end = time.time()

        df["Anomaly"] = labels
        df["Score"] = scores

        anomalies = filter_anomalies(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Anomalies Detected", len(anomalies))
        col3.metric("Execution Time (sec)", round(end-start, 4))

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies)

        plot_pca(scaled_data, labels)
        plot_scores(scores)

        st.subheader("Feature Contribution (Explainability)")
        explanation_df = explain_anomalies(df, selected_cols, labels)
        st.dataframe(explanation_df)

        anomalies.to_excel("anomaly_report.xlsx", index=False)
        with open("anomaly_report.xlsx", "rb") as f:
            st.download_button(
                "Download Anomaly Report",
                f,
                file_name="anomaly_report.xlsx"

            )


