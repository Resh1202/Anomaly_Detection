import pandas as pd

def prepare_download(df):
    return df.to_excel("anomaly_report.xlsx", index=False)

def filter_anomalies(df):
    return df[df["Anomaly"] == -1]