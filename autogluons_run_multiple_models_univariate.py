from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

univariate_data_path = './Datasets/TSB-Forecasting-U/bitcoin_dataset.csv'
univariate_data_path_nn5 = './Datasets/TSB-Forecasting-U/NN5/T1_2.csv'
univariate_data_path_us_gasoline = './Datasets/TSB-Forecasting-U/us_gasoline/us_gasoline.csv'

df = pd.read_csv(univariate_data_path)
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

prediction_length = 10
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    path="autogluon-univariate-forecasting",
    target="target",
    eval_metric="MASE",
    freq="D"  # Adjust to 'H' for hourly or 'W' for weekly data
)

custom_models = {
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

model_names = predictor.model_names()

all_evaluations = {}

for model in model_names:
    evaluation = predictor.evaluate(
        test_data,
        metrics=['MSE', 'RMSE', 'MAE', 'WAPE', 'SMAPE'],
        model=model
    )
    evaluation = {metric: -value for metric, value in evaluation.items()}
    all_evaluations[model] = evaluation

evaluation_df = pd.DataFrame(all_evaluations).transpose()

