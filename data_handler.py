import pandas as pd

def load_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def basic_validation(df):
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    return df
