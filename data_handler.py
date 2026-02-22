import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    return df

def scale_features(df, selected_columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[selected_columns])
    return scaled