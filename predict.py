from config import config 
import utils
import joblib
from model import create_model

def prepare_start_data(val_scale, test_scale, val_cov_scale, test_cov_scale, input_length, output_length, i, using_cov):
    """시작 데이터를 준비하는 함수"""
    if i == 0:
        start_data = val_scale[-input_length:]
        start_cov_data = val_cov_scale[-input_length:] if using_cov else None
    else:
        start_data = val_scale[-(input_length-(output_length*i)):]
        start_test_data = test_scale[:(output_length*i)]
        start_data = start_data.append(start_test_data)
        
        if using_cov:
            start_cov_data = val_cov_scale[-(input_length-(output_length*i)):]
            start_cov_test_data = test_cov_scale[:(output_length*i)]
            start_cov_data = start_cov_data.append(start_cov_test_data)
        else:
            start_cov_data = None
    
    return start_data, start_cov_data 

def predict_chunk(model, start_data, start_cov_data, output_length):
    """단일 청크에 대한 예측을 수행하는 함수"""
    
    return model.predict(n=output_length, series=start_data, past_covariates=start_cov_data)


def handle_remaining_data(val_scale, test_scale, val_cov_scale, test_cov_scale, input_length, remain_data, using_cov):
    """남은 데이터를 처리하는 함수"""
    start_data = val_scale[-remain_data:]
    start_test_data = test_scale[:(input_length-remain_data)]
    start_data = start_data.append(start_test_data)
    
    if using_cov:
        start_cov_data = val_cov_scale[-remain_data:]
        start_test_cov_data = test_cov_scale[:(input_length-remain_data)]
        start_cov_data = start_cov_data.append(start_test_cov_data)
    else:
        start_cov_data = None
    
    return start_data, start_cov_data


def _init_pred_time(model, val_scale, val_cov_scale, test_scale, test_cov_scale, scaler, config):
    """
    test 기간의 t시점~t+input 시점을의 pred를 구하기 위한 초기 데이터 생성
    """
    input = model.input_chunk_length
    output = model.output_chunk_length
    # val 데이터 몇번 반복할지
    repeat_count = input // output
    # 나머지 데이터가 존재한다면
    remain_data = input % output
    pred_all = None
    # 반복해야함
    # 횟수만큼 반복
    for i in range(repeat_count):
        # 횟수가 0이면 val 데이터만 가져옴
        start_data, start_cov_data = prepare_start_data(val_scale, test_scale, val_cov_scale, test_cov_scale, 
                                            input, output, i, config.using_cov)
        pred = predict_chunk(model, start_data, start_cov_data, output)
        pred_all = pred if pred_all is None else pred_all.append(pred)

                
    if repeat_count == 0:
        start_data, start_cov_data = prepare_start_data(val_scale, test_scale, val_cov_scale, test_cov_scale, 
                                                        input, output, 0, config.using_cov)
        pred_all = predict_chunk(model, start_data, start_cov_data, output, config.using_cov)

    if (remain_data != 0) and (input > output):
        start_data, start_cov_data = handle_remaining_data(val_scale, test_scale, val_cov_scale, test_cov_scale, 
                                                           input, remain_data, config.using_cov)
        
        
        pred = predict_chunk(model, start_data, start_cov_data, remain_data, config.using_cov)
        pred_all = pred_all.append(pred) if pred_all is not None else pred
        
    pred_all = scaler.inverse_transform(pred_all)


    return pred_all


def pred_recursive(model, pred_scale, pred_cov_scale, pred_all, scaler, config):
    input = model.input_chunk_length
    output = model.output_chunk_length
    
    # 반복 횟수 확인, 현재 데이터에서 몇번 반복해야하는지 확인
    # pred(t+input_data~t+last) 구하기 위함
    # pred_all(t+1-t+input_data)
    repreat_count = (len(pred_scale) - len(pred_all)) // output
    remain_time = (len(pred_scale) - len(pred_all)) % output

    for i in range(repreat_count):
        start_data = pred_scale[(output*i):(input+(output*i))]
        start_cov_data = pred_cov_scale[(output*i):(input+(output*i))] if config.using_cov else None
        pred = predict_chunk(model, start_data, start_cov_data, output)
        pred_all = pred_all.append(pred)

    if remain_time != 0:
        start_data = pred_scale[-(input+remain_time):-remain_time]
        start_cov_data = pred_cov_scale[-(input+remain_time):-remain_time] if config.using_cov else None
        pred = predict_chunk(model, start_data, start_cov_data, output)
        pred = pred[:remain_time]
        pred_all = pred_all.append(pred)

    pred_all = scaler.inverse_transform(pred_all)
        
    return pred_all




def eval_model(model, val_scale, val_cov_scale=None, test_scale = None, test_cov_scale=None, target_scaler=None, config=config):
    """
    Test 데이터 예측 함수
    """
    model = model.load(f"{config.model_dir}/{config.model_file_name}.pt")
    
    """모델 추론 함수"""
    init_pred = _init_pred_time(model, val_scale, val_cov_scale, test_scale, test_cov_scale, target_scaler, config)
    
    prediction = pred_recursive(model, test_scale, test_cov_scale, init_pred, target_scaler, config)
    
    return prediction


def model_predict(model, target_predict, target_cov=None, config=config):
    """
    모델 예측 함수
    """
    model = model.load(f"{config.model_dir}/{config.model_file_name}.pt")
    output = model.output_chunk_length
    target_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_target_scaler.pkl')

    prediction = model.predict(n = output, series = target_predict, past_covariates=target_cov)
    prediction = target_scaler.inverse_transform(prediction)
        
    return prediction


def __main__():
    model, _ = create_model(config)
    
    val_scale = utils.load_data(config, "target_val.pkl")
    test_scale = utils.load_data(config, "target_test.pkl")
    config.using_cov = True
    config.update_model_name()  
    
    if config.using_cov:
        val_cov_scale = utils.load_data(config, "cov_val.pkl")
        test_cov_scale = utils.load_data(config, "cov_test.pkl")
    else:
        val_cov_scale = None
        test_cov_scale = None
        
    target_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_target_scaler.pkl')
    prediction = eval_model(model, val_scale, val_cov_scale, test_scale[:1000], test_cov_scale, target_scaler, config)
    
    return prediction


if __name__ == "__main__":
    prediction = __main__()