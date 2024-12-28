# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:01:14 2024

@author: TEO
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autogluon.timeseries import TimeSeriesPredictor
from data_parser import get_data_for_chronos_bolt
import pandas as pd

# model_name="chronos-bolt-tiny", mini, small, base
def run_chronos_bolt(data_path, model_name="chronos-bolt-tiny", device="cuda:0", batch_size=16):
    train_data, test_data, scaler = get_data_for_chronos_bolt(data_path)

    prediction_length = 5

    predictor = TimeSeriesPredictor(prediction_length=prediction_length)
    predictor.fit(
        train_data,
        hyperparameters={
            "Chronos": {
                "model_path": f"autogluon/{model_name}",
                "context_length": 64, # this can go up to 2048 for the chronos-bolt models.
                "device": device,
                "batch_size": batch_size,
                "fine_tune": True,
                "fine_tune_steps": 3000,
                "fine_tune_lr": 1e-4
            }
        },
        num_val_windows=1
    )

    predictions = predictor.predict(test_data)

    actual_values = test_data["target"].values.flatten()[:prediction_length]
    median_forecasts = predictions["mean"].values.flatten()

    actual_values = scaler.inverse_transform(actual_values.reshape(-1, 1)).flatten()
    median_forecasts = scaler.inverse_transform(median_forecasts.reshape(-1, 1)).flatten()

    mse = mean_squared_error(actual_values, median_forecasts)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, median_forecasts)
    r2 = r2_score(actual_values, median_forecasts)

    print(f"Model: {model_name}")
    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    return mse, rmse, mae, r2
