from config import config 

def train_model(model, train_data, val_data, train_cov=None, val_cov=None, config=config):
    """모델 학습 함수"""
    model, loss_logging_callback = model
    
    model = model.fit(
            train_data,
            past_covariates=train_cov,
            val_series=val_data,
            val_past_covariates=val_cov
        )

    model.save(f"{config.model_dir}/{config.model_file_name}.pt")
    print(f"{config.model_dir}로 모델 저장완료")
    
    loss_logging_callback.plot_losses()
    return model


def __main__():
    from config import config  
    import utils
    from model import create_model
    model = create_model(config)
    target_train = utils.load_data(config, "target_train.pkl")
    target_val = utils.load_data(config, "target_val.pkl")
    config.using_cov = False
    config.update_model_name()   
     
    if config.using_cov:
        cov_train = utils.load_data(config, "cov_train.pkl")
        cov_val = utils.load_data(config, "cov_val.pkl")
    else:
        cov_train = None
        cov_val = None
    
    model = train_model(model, target_train[0:10000], target_val[0:10000], cov_train, cov_val, config)

    return model


if __name__ == "__main__":
    __main__()
    