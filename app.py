# main_dashboard.py
import streamlit as st
import pandas as pd

from data_loader import load_uploaded_file, basic_validation
from feature_pipeline import scale_features, apply_pca
from detection_models import run_isolation_forest, run_oneclass_svm, run_lof
from explainability_core import get_feature_importance
from evaluation_metrics import show_classification_metrics, plot_roc_curve
from visualization_tools import plot_score_distribution, plot_pca_projection
from config_settings import DEFAULT_CONTAMINATION, DEFAULT_TREES

st.title("FraudLens AI - Explainable Anomaly Detection")

st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Isolation Forest", "One-Class SVM", "LOF"]
)

contamination = st.sidebar.slider(
    "Contamination", 0.001, 0.1, DEFAULT_CONTAMINATION
)

n_estimators = st.sidebar.slider(
    "Number of Trees", 100, 500, DEFAULT_TREES
)

uploaded_file = st.file_uploader("Upload CSV Dataset")

if uploaded_file:
    df = load_uploaded_file(uploaded_file)
    df = basic_validation(df)

    X = df.drop(columns=["Class"], errors="ignore")
    y = df["Class"] if "Class" in df.columns else None

    X_scaled = scale_features(X)

    if model_choice == "Isolation Forest":
        model, scores, labels = run_isolation_forest(
            X_scaled, contamination, n_estimators
        )
    elif model_choice == "One-Class SVM":
        model, scores, labels = run_oneclass_svm(X_scaled, contamination)
    else:
        model, scores, labels = run_lof(X_scaled, contamination)

    df["Anomaly_Label"] = labels

    st.subheader("Results Preview")
    st.write(df.head())

    if scores is not None:
        st.pyplot(plot_score_distribution(scores))

    X_pca = apply_pca(X_scaled)
    st.pyplot(plot_pca_projection(X_pca, labels))

    importance_df = get_feature_importance(model, X.columns)
    if importance_df is not None:
        st.subheader("Feature Contribution")
        st.bar_chart(importance_df.set_index("Feature"))

    if y is not None and scores is not None:
        report, cm = show_classification_metrics(y, labels)
        st.text(report)
        st.write("Confusion Matrix")
        st.write(cm)
        st.pyplot(plot_roc_curve(y, scores))

    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "anomaly_results.csv"
    )
