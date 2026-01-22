import joblib

def detect_model_type(model_path: str):
    model = joblib.load(model_path)

    if hasattr(model, "coef_"):
        return "linear"

    if hasattr(model, "tree_") or hasattr(model, "estimators_"):
        return "tree"

    if hasattr(model, "_fit_X"):
        return "knn"

    return "unknown"
