from sklearn.model_selection import train_test_split
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from data_parser import index_multivariate_data_by_date

multivariate_data_path_air_quality = './Datasets/TSB-Forecasting-M/AirQualityUCI.csv'

pivoted_data = index_multivariate_data_by_date(multivariate_data_path_air_quality)

print("Pivoted Data Columns:\n", pivoted_data.columns)

train_size = 0.8
train_df, test_df = train_test_split(pivoted_data, train_size=train_size, shuffle=False)

train_df = train_df.reset_index()
test_df = test_df.reset_index()

train_df = train_df.melt(id_vars=['date'], var_name='item_id', value_name='value')
test_df = test_df.melt(id_vars=['date'], var_name='item_id', value_name='value')

train_df.rename(columns={'date': 'timestamp'}, inplace=True)
test_df.rename(columns={'date': 'timestamp'}, inplace=True)

train_data = TimeSeriesDataFrame.from_data_frame(
    train_df,
    id_column='item_id',
    timestamp_column='timestamp'
)

test_data = TimeSeriesDataFrame.from_data_frame(
    test_df,
    id_column='item_id',
    timestamp_column='timestamp'
)

predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-multivariate-forecasting",
    target="value",
    eval_metric="MASE",
    freq="H"  # Assuming hourly data
)

custom_models = {
    "DeepAR": {},
    "SimpleFeedForward": {},
}

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
    hyperparameters=custom_models
)

predictions = predictor.predict(test_data)

leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

# TODO: Find a way to get our metrics from these models
# # Calculate custom metrics
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Align predictions and true values
# y_pred = predictions["mean"]
# y_true = test_data["value"]
# y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')

# # Calculate Metrics
# mse = mean_squared_error(y_true_aligned, y_pred_aligned)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
# r2 = r2_score(y_true_aligned, y_pred_aligned)

# print(f"\n--- Evaluation Metrics ---")
# print(f"MSE: {mse:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"R²: {r2:.4f}")
