import pandas as pd
import numpy as np

def explain_anomalies(df, selected_columns, anomaly_labels):

    anomalies = df[anomaly_labels == -1]
    explanations = {}

    for col in selected_columns:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = (anomalies[col] - mean) / std
        explanations[col] = z_scores.abs().mean()

    explanation_df = pd.DataFrame(
        explanations.items(),
        columns=["Feature", "Average Z-Score Contribution"]
    ).sort_values(by="Average Z-Score Contribution", ascending=False)

    return explanation_df