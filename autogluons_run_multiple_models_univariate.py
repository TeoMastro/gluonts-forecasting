from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Paths to datasets
univariate_data_path = './Datasets/TSB-Forecasting-U/bitcoin_dataset.csv'
univariate_data_path_nn5 = './Datasets/TSB-Forecasting-U/NN5/T1_2.csv'
univariate_data_path_us_gasoline = './Datasets/TSB-Forecasting-U/us_gasoline/us_gasoline.csv'

df = pd.read_csv(univariate_data_path)
print(df.head())

df.rename(columns={
    "date": "timestamp",  
    "data": "target"      
}, inplace=True)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["item_id"] = "bitcoin"

train_size = 0.8
train_df, test_df = train_test_split(df, train_size=train_size, shuffle=False)

train_data = TimeSeriesDataFrame.from_data_frame(
    train_df,
    id_column="item_id",
    timestamp_column="timestamp"
)

test_data = TimeSeriesDataFrame.from_data_frame(
    test_df,
    id_column="item_id",
    timestamp_column="timestamp"
)

predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-univariate-forecasting",
    target="target",
    eval_metric="MASE",
    freq="D"  # Adjust to 'H' for hourly or 'W' for weekly data
)

custom_models = {
    "ZeroModel": {},
    "AutoETS": {},
    "AutoCES": {},               
    "Theta": {},                 
    "ADIDA": {},                 
    "Croston": {},                
    "IMAPA": {},                  
    "SimpleFeedForward": {},                  
    "DeepAR": {},                
    "Chronos": {},
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

# TODO: Make sure I have a way to evaluate these models according to ours.
# model_names = predictor.model_names

# results = []

# for model in model_names:
#     print(f"\nEvaluating Model: {model}")
#     predictions = predictor.predict(test_data, model=model)
    
#     y_pred = predictions["mean"]
#     y_true = test_data["target"]

#     y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')

#     mse = mean_squared_error(y_true_aligned, y_pred_aligned)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
#     r2 = r2_score(y_true_aligned, y_pred_aligned)

#     results.append({
#         "Model": model,
#         "MSE": mse,
#         "RMSE": rmse,
#         "MAE": mae,
#         "R²": r2
#     })

#     print(f"MSE: {mse:.4f}")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"R²: {r2:.4f}")

# results_df = pd.DataFrame(results)
# print("\n=== Model Evaluation Results ===")
# print(results_df)
