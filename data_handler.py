import pandas as pd

def load_uploaded_file(uploaded_file):
    return pd.read_csv(uploaded_file)

def basic_validation(df):
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    return df
