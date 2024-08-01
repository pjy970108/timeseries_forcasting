import config

def train_model(model, train_data, val_data, train_cov=None, val_cov=None, config=config):
    """모델 학습 함수"""
    model, loss_logging_callback = model
    
    if config.using_cov:
        model = model.fit(
            train_data,
            past_covariates=[train_cov],
            val_series=val_data,
            val_past_covariates=[val_cov]
        )
    else:
        model = model.fit(
            train_data,
            val_series=val_data,
        )

    model.save(f"{config.model_dir}/{config.model_name}.pt")
    print(f"{config.model_dir}로 모델 저장완료")
    
    loss_logging_callback.plot_losses()
    return model


def __main__():
    import config
    import utils
    from model import create_model
    model = create_model(config)
    target_train = utils.load_data(config.data_dir, "target_train.pkl")
    target_val = utils.load_data(config.data_dir, "target_val.pkl")
    if config.using_cov:
        cov_train = utils.load_data(config.data_dir, "cov_train.pkl")
        cov_val = utils.load_data(config.data_dir, "cov_val.pkl")
    else:
        cov_train = None
        cov_val = None
    
    model = train_model(model, target_train[0:10000], target_val[0:10000], cov_train, cov_val, config)

    return model

if __name__ == "__main__":
    __main__()
    