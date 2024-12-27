# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:04:02 2024

@author: TEO
"""

from model_wrapper import *

univariate_data_path = './Datasets/TSB-Forecasting-U/bitcoin_dataset.csv'
univariate_data_path_nn5 = './Datasets/TSB-Forecasting-U/NN5/T1_2.csv'
univariate_data_path_us_gasoline = './Datasets/TSB-Forecasting-U/us_gasoline/us_gasoline.csv'
multivariate_data_path = './Datasets/TSB-Forecasting-M/AQWan.csv'
multivariate_data_path_air_quality = './Datasets/TSB-Forecasting-M/AirQualityUCI.csv'
multivariate_data_path_beiijing_multisite_air_quality = './Datasets/TSB-Forecasting-M/beijing_multisite_air_quality/PRSA_Data_Aotizhongxin_20130301-20170228.csv'
multivariate_data_path_australian_tourism = './Datasets/TSB-Forecasting-M/australian_tourism/australian_tourism.csv'

target_col = 'PM2.5'
run_chronos_bolt(univariate_data_path)