from make_feature import get_features
import os
from preprocessing import main_preprocessing
from model import create_model
from train import train_model
from predict import eval_model, model_predict
import config
import utils
import joblib

def main(config):
    # 만약 저장된 된 데이터가 있으면 불러오고 없으면 생성
    if not config.features_file in os.listdir(config.data_dir):
        data = get_features(config)
        
    else:
        print("파일이 이미 존재합니다.")
        
    # 데이터 전처리
    # 데이터가 있다면 데이터를 불러옵니다. 공변량을 사용하지 않으면 cov 파일은 None으로 반환됩니다.
    if any("train" in file for file in os.listdir(config.data_dir)):
        target_train = utils.load_data(config.data_dir, "target_train.pkl")
        target_val = utils.load_data(config.data_dir, "target_val.pkl")
        target_test = utils.load_data(config.data_dir, "target_test.pkl")
        cov_train = utils.load_data(config.data_dir, "cov_train.pkl")
        cov_val = utils.load_data(config.data_dir, "cov_val.pkl")
        cov_test = utils.load_data(config.data_dir, "cov_test.pkl")
        target_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_target_scaler.pkl')
        cov_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_cov_scaler.pkl')
    
    else: #없다면 train, test, val 생성 및 저장
        target_train, target_val, target_test, target_scaler, cov_train, cov_val, cov_test, cov_scaler = main_preprocessing(config)
    
    if config.mode == "train":
        # 모델 생성
        model = create_model(config)
        # 모델 학습
        model = train_model(model, target_train[0:10000], target_val[0:2000], cov_train[0:10000], cov_val[0:2000], config)
        return model
    
    elif config.mode == "predict":
        model, _ = create_model(config)

        if config.eval_pred: # 평가를 위함
            # 설정된 모델을 불러옴
            # 모델 예측
            prediction = eval_model(model, target_val, cov_val, target_test[0:1000], cov_test[0:1000], target_scaler, config)

        else: # 예측만을 위함
            prediction = model_predict(model, target_test, cov_test, config)
        
        return prediction    


if __name__ == "__main__":
    config.mode = "train"
    model = main(config)
    
    config.mode = "predict"
    config.eval_pred = True
    prediction = main(config)
