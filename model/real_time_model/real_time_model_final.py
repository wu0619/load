import numpy as np
import xgboost as xgb
from joblib import load
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error


class Model:
    """
    Expected data shape:
    Shape X: (n_Data, 86) 
    -> past 38 steps + last week 38 steps + last week future 10 steps = 86 steps
    Shape Y: (n_Data, 10)
    -> predict future 2.5 hours
    """

    def __init__(self) -> None:
        self.scaler = load("scaler.joblib")

        self.svr = load("svr0.joblib")
        self.xgboost = xgb.XGBRegressor()
        self.xgboost.load_model('xgboost0.model')

        self.y = None

    def predict(self, X: np.ndarray):
        self.y = np.empty(shape=(X.shape[0], 96))

        if not isinstance(X, np.ndarray):
            raise TypeError("Expected X to be a NDArray.")

        if X.ndim != 2:
            raise ValueError(f"n_dim of X must be 2, but get {X.ndim}.")

        X_scaled = self.scaler.transform(X)
        X_filtered = np.array(savgol_filter(X_scaled, 10, 4))
        X_residual = X_scaled - X_filtered

        pred_svr = self.svr0.predict(X_filtered)
        pred_xgboost = self.xgboost0.predict(X_residual)
        self.y = pred_svr + pred_xgboost

        return self.y

def main():
    pass

if __name__=='__main__':
    main()