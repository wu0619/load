import numpy as np
import pandas as pd
import load_model
import time
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_absolute_percentage_error
import load_model.load_model_api

def generate_sets_for_all_timestamps(data):
    load_col = 'out.site_energy.total.energy_consumption.kwh'
    start = data['timestamp'].min() + DateOffset(days=3)
    end = data['timestamp'].max() - DateOffset(minutes=96*15)
    timestamps = pd.date_range(start=start, end=end, freq='15min')

    training_sets = []
    target_sets = []
    for timestamp in timestamps:
        # Calculate the range for the current period's data
        start_time_current = timestamp - DateOffset(days=2, hours=23, minutes=45)
        end_time_current = timestamp

        # Calculate the target range (the next 10 steps after the current timestamp)
        target_start_time = timestamp + DateOffset(minutes=15)
        target_end_time = timestamp + DateOffset(minutes=15*96) 

        # Filter the data for training and target sets
        current_data = data[(data['timestamp'] >= start_time_current) & (data['timestamp'] <= end_time_current)]
        target_data = data[(data['timestamp'] >= target_start_time) & (data['timestamp'] <= target_end_time)]

        # Combine current and last week data for the training set
        training_data = pd.concat([current_data]).reset_index(drop=True)
        # Save the training and target sets
        if not training_data.empty and not target_data.empty:
            training_sets.append(training_data[load_col])
            target_sets.append(target_data[load_col])
        break

    training_sets = np.array(training_sets)
    target_sets = np.array(target_sets)

    return training_sets, target_sets,

def main():
    # dirs
    DATA_DIR = "./load.csv"
    load_col = 'out.site_energy.total.energy_consumption.kwh'
    data = pd.read_csv(DATA_DIR)

    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # kWh -> MW
    data[load_col] = data[load_col] * 4 / 1e3
    # data[load_col] = np.arange(len(data[load_col]))

    print("sliding window ...")
    training_sets, target_sets = generate_sets_for_all_timestamps(data)
    model = load_model.load_model_api.LoadModel(modelPath="./load_model.keras", scalerPath="./scaler.pkl")
    start_time = time.time()
    result = model.predict(data=training_sets[0])
    end_time = time.time()
    print(mean_absolute_percentage_error(target_sets, result))
    elapsed_time = end_time - start_time
    print(result.shape)
    print(f"Function execution took {elapsed_time} seconds.")
    # took 1.96 ~ 2.13s
if __name__ == '__main__':
    main()