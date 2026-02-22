from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def detect_anomalies(model_name, data, contamination=0.05):

    if model_name == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42)
        labels = model.fit_predict(data)
        scores = model.decision_function(data)

    elif model_name == "Local Outlier Factor":
        model = LocalOutlierFactor(contamination=contamination)
        labels = model.fit_predict(data)
        scores = -model.negative_outlier_factor_

    elif model_name == "One-Class SVM":
        model = OneClassSVM(nu=contamination)
        labels = model.fit_predict(data)
        scores = model.decision_function(data)

    return labels, scores