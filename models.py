from sklearn.ensemble import IsolationForest

def detect_anomalies(model_name, data, contamination=0.05):

    if model_name == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42)
        labels = model.fit_predict(data)
        scores = model.decision_function(data)

    return labels, scores
