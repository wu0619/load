import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

class Model:
    """
    Expected data shape:
    Shape X: (n_Data, 384)
    Shape Y: (n_Data, 96)
    """

    def __init__(self) -> None:

        self.scaler = load("scaler.joblib")
        self.kmeans = load("kmeans.joblib")

        self.svr0 = load("svr0.joblib")
        self.svr1 = load("svr1.joblib")
        self.svr2 = load("svr2.joblib")
        self.svr3 = load("svr3.joblib")

        self.xgboost0 = xgb.XGBRegressor()
        self.xgboost1 = xgb.XGBRegressor()
        self.xgboost2 = xgb.XGBRegressor()
        self.xgboost3 = xgb.XGBRegressor()

        self.xgboost0.load_model('xgboost0.model')
        self.xgboost1.load_model('xgboost1.model')
        self.xgboost2.load_model('xgboost2.model')
        self.xgboost3.load_model('xgboost3.model')

        self.K_label = None
        self.X = None
        self.y = None

    def predict(self, X: list):
        self.X = np.array(X)
        self.y = np.empty(shape=(self.X.shape[0], 96))

        X_scaled = self.scaler.transform(self.X.reshape(-1, 1)).reshape(1, -1)
        X_label = self.kmeans.predict(X_scaled)

        X_filtered = np.array(savgol_filter(X_scaled, 10, 4))
        X_residual = X_scaled - X_filtered

        if X_label == 0:
            pred_svr = self.svr0.predict(X_filtered)
            pred_xgboost = self.xgboost0.predict(X_residual)
            y_pred = pred_svr + pred_xgboost
            self.y = self.scaler.inverse_transform(y_pred)
        elif X_label == 1:
            pred_svr = self.svr1.predict(X_filtered)
            pred_xgboost = self.xgboost1.predict(X_residual)
            y_pred = pred_svr + pred_xgboost
            self.y = self.scaler.inverse_transform(y_pred)
        elif X_label == 2:
            pred_svr = self.svr2.predict(X_filtered)
            pred_xgboost = self.xgboost2.predict(X_residual)
            y_pred = pred_svr + pred_xgboost
            self.y = self.scaler.inverse_transform(y_pred)
        elif X_label == 3:
            pred_svr = self.svr3.predict(X_filtered)
            pred_xgboost = self.xgboost3.predict(X_residual)
            y_pred = pred_svr + pred_xgboost
            self.y = self.scaler.inverse_transform(y_pred)
        return self.y[0]
