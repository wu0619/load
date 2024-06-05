import numpy as np
import pandas as pd
import math
import seaborn as sns
from pandas.tseries.offsets import DateOffset

class TimeSeriesImageCoder():
    def __init__(
            self,
        X: pd.DataFrame,
        n_predict: int,
        height: int,
        width: int,
        n_days: int,
        n_window_shift: str
    ) -> None:
        self.X = X
        self.h = height
        self.m = width
        self.d_b = n_days
        self.shift = n_window_shift
        self.n_predict = n_predict
        self.Lb = self.h * self.m
        self.Ls = math.ceil(self.n_predict / self.m) * self.m
        self.timestamps = self.generate_timestamps()
        print(f"Lb: {self.Lb}")
        print(f"Ls: {self.Ls}")

    def generate_timestamps(self):
        start = self.X['Timestamp'].min() + DateOffset(days=3)
        end = self.X['Timestamp'].max() - DateOffset(minutes=96*15)
        timestamps = pd.date_range(start=start, end=end, freq=self.shift)
        return timestamps
    
    def __symmetric_ud(self, vector):
        symmetric_matrices = []
    
        for i in range(0, self.d_b*96, 96):
            day_data = vector[i:i+96]
            day_data_reshaped = np.reshape(day_data, (self.h, self.m))
            symmetric_matrix = np.vstack((day_data_reshaped, np.flipud(day_data_reshaped)))
            symmetric_matrices.append(symmetric_matrix)
        
        return symmetric_matrices

    def __symmetric_lr(self, vector):
        symmetric_matrices = []
    
        for i in range(0, self.d_b*96, 96):
            day_data = vector[i:i+96]
            day_data_reshaped = np.reshape(day_data, (self.h, self.m))
            symmetric_matrix = np.hstack((day_data_reshaped, np.fliplr(day_data_reshaped)))
            symmetric_matrices.append(symmetric_matrix)
        
        return symmetric_matrices


    def encode_b(self):
        training_sets = []
        target_sets = []
        self.X_timeseries_flatten = []
        self.X_timestamp = []
        self.y_timestamp = []
        for steps in self.timestamps:
            training_start_b = steps - DateOffset(days=self.d_b-1, hours=23, minutes=45)
            training_end = steps
            target_start = training_end + DateOffset(minutes=15)
            target_end = steps + DateOffset(minutes=(self.n_predict)*15)
            training_data = self.X[(self.X['Timestamp'] >= training_start_b) & (self.X['Timestamp'] <= training_end)]
            target_data = self.X[(self.X['Timestamp'] >= target_start) & (self.X['Timestamp'] <= target_end)]
            if not training_data.empty and not target_data.empty:
                self.X_timeseries_flatten.append(training_data['Load'])
                self.X_timestamp.append(training_data['Timestamp'])
                self.y_timestamp.append(target_data['Timestamp'])
                # training_reshaped = np.array(training_data['Load']).reshape(self.d_b, self.h, self.m)
                training_reshaped = self.__symmetric_ud(training_data['Load'])
                training_sets.append(training_reshaped)
                target_reshaped = np.array(target_data['Load']).reshape(math.ceil(self.n_predict/self.m), min(self.n_predict, self.m))
                target_sets.append(target_reshaped.flatten())
        training_sets = np.array(training_sets)
        target_sets = np.array(target_sets)

        self.X_timeseries_flatten = np.array(self.X_timeseries_flatten)
        self.X_timestamp = np.array(self.X_timestamp)
        self.y_timestamp = np.array(self.y_timestamp)
        return training_sets, target_sets
    
    def encode(self):
        training_sets_b, target_sets = self.encode_b()
        training_sets_b = np.transpose(training_sets_b, (0, 2, 3, 1))
        return training_sets_b, target_sets
    
    
if __name__ == '__main__':
    """
    !! parameter settings
    n_predict: predict steps
    height: final height of the image:
                height * 2 if the n_predict <= width,
                height * 2 + 1 if the n_predict > width
    width: width of the image
    n_days: use past n days historical time series data as input (number of channel)
    n_window_shift: the shift interval of sliding window
    """
    n_predict = 2
    height = 8
    width = 12
    n_days = 3
    n_window_shift = "15min"

    # import your data here ...
    # Timestamp / Load
    FILE_DIR = './data.csv'
    data = pd.read_csv(FILE_DIR)

    encoder = TimeSeriesImageCoder(
        X=data,
        n_predict=n_predict,
        height=height,
        width=width,
        n_days=n_days,
        n_window_shift=n_window_shift
    )
    encoded_X, encoded_y = encoder.encode()