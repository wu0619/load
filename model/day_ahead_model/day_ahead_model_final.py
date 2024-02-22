import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error


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
        self.y = None

    def predict(self, X: np.ndarray):
        self.y = np.empty(shape=(X.shape[0], 96))

        if not isinstance(X, np.ndarray):
            raise TypeError("Expected X to be a NDArray.")

        if X.ndim != 2:
            raise ValueError(f"n_dim of X must be 2, but get {X.ndim}.")

        X_scaled = self.scaler.transform(X)
        X_label = self.kmeans.predict(X_scaled)
        k0 = np.where(X_label == 0)
        k1 = np.where(X_label == 1)
        k2 = np.where(X_label == 2)
        k3 = np.where(X_label == 3)

        X_0 = X_scaled[k0]
        X_1 = X_scaled[k1]
        X_2 = X_scaled[k2]
        X_3 = X_scaled[k3]

        X_filtered_0 = np.array(savgol_filter(X_0, 10, 4))
        X_filtered_1 = np.array(savgol_filter(X_1, 10, 4))
        X_filtered_2 = np.array(savgol_filter(X_2, 10, 4))
        X_filtered_3 = np.array(savgol_filter(X_3, 10, 4))

        X_residual_0 = X_0 - X_filtered_0
        X_residual_1 = X_1 - X_filtered_1
        X_residual_2 = X_2 - X_filtered_2
        X_residual_3 = X_3 - X_filtered_3

        pred_svr0 = self.svr0.predict(X_filtered_0)
        pred_svr1 = self.svr1.predict(X_filtered_1)
        pred_svr2 = self.svr2.predict(X_filtered_2)
        pred_svr3 = self.svr3.predict(X_filtered_3)

        pred_xgboost0 = self.xgboost0.predict(X_residual_0)
        pred_xgboost1 = self.xgboost1.predict(X_residual_1)
        pred_xgboost2 = self.xgboost2.predict(X_residual_2)
        pred_xgboost3 = self.xgboost3.predict(X_residual_3)

        y_pred_0 = pred_svr0 + pred_xgboost0
        y_pred_1 = pred_svr1 + pred_xgboost1
        y_pred_2 = pred_svr2 + pred_xgboost2
        y_pred_3 = pred_svr3 + pred_xgboost3

        self.y[k0] = y_pred_0
        self.y[k1] = y_pred_1
        self.y[k2] = y_pred_2
        self.y[k3] = y_pred_3

        return self.y

def main():
    X = pd.read_csv('testing.csv', header=None, index_col=False).to_numpy()
    y = pd.read_csv('ans.csv', header=None, index_col=False).to_numpy()
    model = Model()
    pred = model.predict(X)
    print(pred)
    mse = mean_squared_error(pred, y)
    print('mse:', mse)

if __name__=='__main__':
    main()