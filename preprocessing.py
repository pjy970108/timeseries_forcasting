from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, BaseDataTransformer
from darts.utils.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, maxabs_scale, minmax_scale
from utils import load_data, save_data
import joblib
from config import config 

    
class Fill_value(BaseDataTransformer):
    @staticmethod
    def ts_transform(series, params, **kwargs):
        interpolation_method = params["interpolation_method"]
        df = series.pd_dataframe()
        # 보간 방법에 따른 변환 로직
        if interpolation_method == 'linear':
            df_interpolated = df.interpolate(method='linear')
        elif interpolation_method == 'polynomial':
            order = params['fixed'].get('order', 2)
            df_interpolated = df.interpolate(method='polynomial', order=order)
        elif interpolation_method == 'spline':
            order = params['fixed'].get('order', 3)
            df_interpolated = df.interpolate(method='spline', order=order)
        elif interpolation_method == 'nearest':
            df_interpolated = df.interpolate(method='nearest')
        elif interpolation_method == 'time':
            df_interpolated = df.interpolate(method='time')
        elif interpolation_method == 'cubic':
            df_interpolated = df.interpolate(method='cubic')
        elif interpolation_method == 'quadratic':
            df_interpolated = df.interpolate(method='quadratic')
        elif interpolation_method == 'pad':
            df_interpolated = df.interpolate(method='pad')
        elif interpolation_method == 'bfill':
            df_interpolated = df.interpolate(method='bfill')
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")
        
        # 변환된 DataFrame을 TimeSeries로 변환
        transformed_series = TimeSeries.from_dataframe(df_interpolated)
        
        # 변환된 시계열 반환
        return transformed_series


def make_timeseries(config, data):
    # 만약 경로에 target_column_features.pkl 파일이 있다면 해당 파일을 로드
    try:
        df = load_data(config.data_dir, f"{config.target_column}_features.pkl")
    except:
        # 데이터가 없다면 새로 만들어서 저장
        df = data

    fill_value = Fill_value()
    # 만약 공변량 데이터를 사용한다면    
    if config.using_cov == True:
        config.time_col = config.time_col
    
        target_df = TimeSeries.from_dataframe(df, config.time_col, "target", freq=config.freq)
        
        cov_df = df[config.cov_cols]
        cov_df = TimeSeries.from_dataframe(cov_df, config.time_col, freq=config.freq)
    
    
        target_df = fill_value.ts_transform(target_df, config.fill_params)
        cov_df = fill_value.ts_transform(cov_df, config.fill_params)
    
        return target_df, cov_df
    else: # 공변량 데이터를 사용하지 않는다면
        target_df = TimeSeries.from_dataframe(df, config.time_col, "target", freq=config.freq)
        target_df = fill_value.ts_transform(target_df, config.fill_params)
        
        return target_df, None

def split_timeseries(target_df, cov_df=None, config=config):
    if cov_df is None:
        target_train, target_test = train_test_split(target_df, test_size=config.test_split)
        target_train, target_val = train_test_split(target_train, test_size=config.val_split)
    
        return target_train, target_val, target_test, None, None, None
    
    else:
        target_train, target_test = train_test_split(target_df, test_size=config.test_split)
        target_train, target_val = train_test_split(target_train, test_size=config.val_split)
    
        cov_train, cov_test = train_test_split(cov_df, test_size=config.test_split)
        cov_train, cov_val = train_test_split(cov_train, test_size=config.val_split)
    
        return target_train, target_val, target_test, cov_train, cov_val, cov_test


def fit_scale(series, scaler):
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'maxabs':
        scaler = maxabs_scale()
    elif scaler == 'minmax':
        scaler = minmax_scale()
    
    scaler_model = Scaler(scaler)
    series = scaler_model.fit_transform(series)
    return series, scaler_model


def transfrom(series, scaler_model):
    series = scaler_model.transform(series)
    return series


def scale_timeseries(target_train, target_val, target_test, cov_train, cov_val, cov_test, config):
    # target_df, cov_df를 스케일링
    # 스케일링은 target_df만 진행
    target_train, target_scaler = fit_scale(target_train, config.scaler)
    target_val = transfrom(target_val, target_scaler)
    target_test = transfrom(target_test, target_scaler)
    if cov_train is None:
        return target_train, target_val, target_test, target_scaler, None, None, None, None
    else:
        cov_train, cov_scaler = fit_scale(cov_train, config.scaler)
        cov_val = transfrom(cov_val, cov_scaler)
        cov_test = transfrom(cov_test, cov_scaler)
             
    return target_train, target_val, target_test, target_scaler, cov_train, cov_val, cov_test, cov_scaler


def main_preprocessing(config, data):

    # 데이터를 불러오고, 결측치를 채움
    target_df, cov_df = make_timeseries(config, data)
    # 데이터를 train, val, test로 나눔        
    target_train, target_val, target_test, cov_train, cov_val, cov_test = split_timeseries(target_df, cov_df, config)
    save_data(target_train, config, "not_norm_target_train.pkl")
    save_data(target_val, config, "not_norm_target_val.pkl")
    save_data(target_val, config, "not_norm_target_test.pkl")
    if cov_train is not None:
        save_data(cov_train, config, "not_norm_cov_train.pkl")
        save_data(cov_val, config, "not_norm_cov_val.pkl")
        save_data(cov_test, config, "not_norm_cov_test.pkl")
    # 데이터를 스케일링
    target_train, target_val, target_test, target_scaler, cov_train, cov_val, cov_test, cov_scaler = scale_timeseries(target_train, target_val, target_test, cov_train, cov_val, cov_test, config)
    
    if config.save_data == True:
        if cov_train is not None:
            save_data(target_train, config, "target_train.pkl")
            save_data(target_val, config, "target_val.pkl")
            save_data(target_test, config, "target_test.pkl")
            save_data(cov_train, config, "cov_train.pkl")
            save_data(cov_val, config, "cov_val.pkl")
            save_data(cov_test, config, "cov_test.pkl")
            joblib.dump(target_scaler, f'{config.data_dir}/{config.scaler}_target_scaler.pkl')
            joblib.dump(cov_scaler, f'{config.data_dir}/{config.scaler}_cov_scaler.pkl')
            print("데이터 저장 완료")
        else:
            save_data(target_train, config, "target_train.pkl")
            save_data(target_val, config, "target_val.pkl")
            save_data(target_test, config, "target_test.pkl")
            joblib.dump(target_scaler, f'{config.data_dir}/{config.scaler}_target_scaler.pkl')
            print("데이터 저장 완료")  
    
    return target_train, target_val, target_test, target_scaler, cov_train, cov_val, cov_test, cov_scaler
        

def __main__():
    import utils
    data = utils.load_data(config.data_dir, f"{config.target_column}_features.pkl")
    config.using_cov = True
    target_train, target_val, target_test, cov_train, cov_val, cov_test, target_scaler, cov_scaler = main_preprocessing(config, data)

if __name__ == "__main__":
    __main__()
