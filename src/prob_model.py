import numpy as np

from sklearn.linear_model import LogisticRegressionCV

def fit_model(
        features : np.ndarray,
        labels : np.ndarray, 
        model_type : str, 
        loss : str = None
):
    if model_type == "logistic":
        model = LogisticRegressionCV()
        model.fit(X=features, y=labels)
        return model
    elif model_type == "XGBoost":
        raise ValueError("not implemented yet")
    else:
        return ValueError(f"{model_type} not available.")

    