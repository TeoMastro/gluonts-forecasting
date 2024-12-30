from sklearn.model_selection import train_test_split
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from data_parser import index_multivariate_data_by_date

multivariate_data_path_air_quality = './Datasets/TSB-Forecasting-M/AirQualityUCI.csv'

pivoted_data = index_multivariate_data_by_date(multivariate_data_path_air_quality)

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

prediction_length = 10
predictor = TimeSeriesPredictor(
    prediction_length=10,
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