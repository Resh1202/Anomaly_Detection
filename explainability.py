# explainability_core.py
import pandas as pd

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": abs(importance)
        }).sort_values(by="Importance", ascending=False)
        return df
    else:
        return None
