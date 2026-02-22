from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from config_settings import RANDOM_STATE

def run_isolation_forest(X, contamination, n_estimators):
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=RANDOM_STATE
    )
    model.fit(X)
    scores = model.decision_function(X)
    labels = model.predict(X)
    return model, scores, labels

def run_oneclass_svm(X, contamination):
    model = OneClassSVM(nu=contamination)
    labels = model.fit_predict(X)
    return model, None, labels

def run_lof(X, contamination):
    model = LocalOutlierFactor(contamination=contamination)
    labels = model.fit_predict(X)
    return model, None, labels
