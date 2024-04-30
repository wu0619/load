import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

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

        self.svr = load("svr.joblib")
        self.xgboost = xgb.XGBRegressor()
        self.xgboost.load_model('xgboost.model')

        self.X = None
        self.y = None

    def predict(self, X: list):
        self.X = np.array(X)
        self.y = np.empty(shape=(self.X.shape[0], 10))

        X_scaled = self.scaler.transform(self.X.reshape(-1, 1)).reshape(1, -1)
        X_filtered = np.array(savgol_filter(X_scaled, 10, 4))
        X_residual = X_scaled - X_filtered

        pred_svr = self.svr.predict(X_filtered)
        pred_xgboost = self.xgboost.predict(X_residual)
        self.y = pred_svr + pred_xgboost
        self.y = self.scaler.inverse_transform(self.y)
        return self.y[0]