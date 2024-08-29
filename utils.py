## 데이터를 저장하고 불러오는 함수들이 정의되어 있음
## 모델을 저장하고 불러오는 함수들이 정의되어있음

import pandas as pd
import torch
from darts import TimeSeries
from config import config 
import os
import joblib


def csv_data(config, file_name):
    """csv 파일을 로드하는 함수"""
    return pd.read_csv(f"{config.data_dir}/{file_name}", index_col=0)


def load_data(config, file_name):
    """데이터를 로드하는 함수"""
    file_path = os.path.join(config.data_dir, file_name)
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        return None

def save_data(data, config, file_name):
    """데이터를 저장하는 함수"""
    
    data.to_pickle(f"{config.data_dir}/{file_name}")
    print(f"complete_save_features_file")

def create_timeseries(df, time_col, target_col, freq='1min'):
    """DataFrame을 TimeSeries로 변환하는 함수"""
    return TimeSeries.from_dataframe(df, time_col, target_col, freq=freq)

def load_all_data(config):
    """모든 데이터를 로드하는 함수"""
    if config.using_cov == True:
        target_train = load_data(config, "target_train.pkl")
        target_val = load_data(config, "target_val.pkl")
        target_test = load_data(config, "target_test.pkl")
        cov_train = load_data(config, "cov_train.pkl")
        cov_val = load_data(config, "cov_val.pkl")
        cov_test = load_data(config, "cov_test.pkl")
        target_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_target_scaler.pkl')
        cov_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_cov_scaler.pkl')
        return target_train, target_val, target_test, target_scaler, cov_train, cov_val, cov_test, cov_scaler
    else:
        target_train = load_data(config, "target_train.pkl")
        target_val = load_data(config, "target_val.pkl")
        target_test = load_data(config, "target_test.pkl")
        target_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_target_scaler.pkl')
        return target_train, target_val, target_test, target_scaler, None, None, None, None
    