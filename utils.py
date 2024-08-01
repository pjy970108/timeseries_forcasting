## 데이터를 저장하고 불러오는 함수들이 정의되어 있음
## 모델을 저장하고 불러오는 함수들이 정의되어있음

import pandas as pd
import torch
from darts import TimeSeries
from config import data_dir, model_dir
import os

def csv_data(data_dir, file_name):
    """csv 파일을 로드하는 함수"""
    return pd.read_csv(f"{data_dir}/{file_name}", index_col=0)


def load_data(data_dir, file_name):
    """데이터를 로드하는 함수"""
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        return None

def save_data(data, data_dir, file_name):
    """데이터를 저장하는 함수"""
    data.to_pickle(f"{data_dir}/{file_name}")
    print(f"complete_save_features_file")

def create_timeseries(df, time_col, target_col, freq='1min'):
    """DataFrame을 TimeSeries로 변환하는 함수"""
    return TimeSeries.from_dataframe(df, time_col, target_col, freq=freq)