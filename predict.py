import config
import utils
import joblib
from model import create_model

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
    # 반복해야함
    # 횟수만큼 반복
    # 공변량 사용시 
    if config.using_cov:
        for i in range(repeat_count):
            # 횟수가 0이면 val 데이터만 가져옴
            if i == 0:
                start_data = val_scale[-input:]
                start_cov_data = val_cov_scale[-input:]
                pred = model.predict(n = output, series = start_data, past_covariates=[start_cov_data])
                pred_all = pred
            else:
                start_data = val_scale[-(input-(output*i)):]
                start_test_data = test_scale[:(output*i)]
                start_data = start_data.append(start_test_data)
                
                start_cov_data = val_cov_scale[-(input-(output*i)):]
                start_cov_test_data = test_cov_scale[:(output*i)]
                start_cov_data = start_cov_data.append(start_cov_test_data)
                
                pred = model.predict(n = output, series = start_data, past_covariates=[start_cov_data])

                pred_all = pred_all.append(pred)
                
        if repeat_count == 0:
            start_data = val_scale[-input:]
            start_cov_data = val_cov_scale[-input:]
            pred = model.predict(n = output, series = start_data, past_covariates=[start_cov_data])
            pred_all = pred

        if (remain_data != 0) & (input > output):
            start_data = val_scale[-remain_data:]
            start_test_data = test_scale[:(input-remain_data)]
            start_cov_data = val_cov_scale[-remain_data:]
            start_test_cov_data = test_cov_scale[:(input-remain_data)]
            start_data = start_data.append(start_test_data)
            start_cov_data = start_cov_data.append(start_test_cov_data)          
            pred = model.predict(n = remain_data, series = start_data, past_covariates=[start_cov_data])
            pred_all = pred_all.append(pred)
    # 공변량 사용하지 않을 때
    else:
        for i in range(repeat_count):
            # 횟수가 0이면 val 데이터만 가져옴
            if i == 0:
                start_data = val_scale[-input:]
                pred = model.predict(n = output, series = start_data)
                pred_all = pred
            else:
                start_data = val_scale[-(input-(output*i)):]
                start_test_data = test_scale[:(output*i)]
                start_data = start_data.append(start_test_data)
                pred = model.predict(n = output, series = start_data)
                pred_all = pred_all.append(pred)
                
        if repeat_count == 0:
            start_data = val_scale[-input:]
            pred = model.predict(n = output, series = start_data)
            pred_all = pred

        if (remain_data != 0) & (input > output):
            start_data = val_scale[-remain_data:]
            start_test_data = test_scale[:(input-remain_data)]
            start_data = start_data.append(start_test_data)
            pred = model.predict(n = remain_data, series = start_data)
            pred_all = pred_all.append(pred)
        
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
    if config.using_cov:
        for i in range(repreat_count):
            if i == 0:
                start_data = pred_scale[:input]
                start_cov_data = pred_cov_scale[:input]
                pred = model.predict(n = output, series = start_data, past_covariates=[start_cov_data])
                pred_all = pred_all.append(pred)
            else:
                start_data = pred_scale[(output*i):(input+(output*i))]
                start_cov_data = pred_cov_scale[(output*i):(input+(output*i))]
                pred = model.predict(n = output, series = start_data, past_covariates=[start_cov_data])
                pred_all = pred_all.append(pred)
                
        if (remain_time != 0):
            start_data = pred_scale[-(input+remain_time):-remain_time]
            start_cov_data = pred_cov_scale[-(input+remain_time):-remain_time]
            pred = model.predict(n = output, series = start_data, past_covariates=[start_cov_data])
            pred = pred[:remain_time]
            pred_all = pred_all.append(pred)
    else:
        for i in range(repreat_count):
            if i == 0:
                start_data = pred_scale[:input]
                pred = model.predict(n = output, series = start_data)
                pred_all = pred_all.append(pred)
            else:
                start_data = pred_scale[(output*i):(input+(output*i))]
                pred = model.predict(n = output, series = start_data)
                pred_all = pred_all.append(pred)  
                
        if (remain_time != 0):
            start_data = pred_scale[-(input+remain_time):-remain_time]
            pred = model.predict(n = output, series = start_data)
            pred = pred[:remain_time]
            pred_all = pred_all.append(pred)
        
    pred_all = scaler.inverse_transform(pred_all)
        
    return pred_all




def eval_model(model, val_scale, val_cov_scale=None, test_scale = None, test_cov_scale=None, target_scaler=None, config=config):
    """
    Test 데이터 예측 함수
    """
    model = model.load(f"{config.model_dir}/{config.model_name}.pt")
    
    """모델 추론 함수"""
    if config.using_cov:
        init_pred = _init_pred_time(model, val_scale, val_cov_scale, test_scale, test_cov_scale, target_scaler, config)
        prediction = pred_recursive(model, test_scale, test_cov_scale, init_pred, target_scaler, config)

    else:
        init_pred = _init_pred_time(model, val_scale, val_cov_scale, test_scale, test_cov_scale, target_scaler, config)
        prediction = pred_recursive(model, test_scale, test_cov_scale, init_pred, target_scaler, config)

    
    return prediction


def model_predict(model, target_predict, target_cov=None, config=config):
    """
    모델 예측 함수
    """
    model = model.load(f"{config.model_dir}/{config.model_name}.pt")
    output = model.output_chunk_length
    target_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_target_scaler.pkl')

    if config.using_cov:
        prediction = model.predict(n = output, series = target_predict, past_covariates=[target_cov])
        prediction = target_scaler.inverse_transform(prediction)
        
    else:
        prediction = model.predict(n = output, series = target_predict)
        prediction = target_scaler.inverse_transform(prediction)

    return prediction


def __main__():
    model = create_model(config)
    val_scale = utils.load_data(config.data_dir, "target_val.pkl")
    test_scale = utils.load_data(config.data_dir, "target_test.pkl")
    if config.using_cov:
        val_cov_scale = utils.load_data(config.data_dir, "cov_val.pkl")
        test_cov_scale = utils.load_data(config.data_dir, "cov_test.pkl")
    else:
        val_cov_scale = None
        test_cov_scale = None
        
    target_scaler = joblib.load(f'{config.data_dir}/{config.scaler}_target_scaler.pkl')
    prediction = eval_model(model, val_scale, val_cov_scale, test_scale, test_cov_scale, target_scaler, config)
    return prediction


if __name__ == "__main__":
    prediction = __main__()