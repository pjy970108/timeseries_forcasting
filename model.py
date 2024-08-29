from config import config
from darts.models import DLinearModel, NLinearModel, TiDEModel, TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from visualize import LossLoggingCallback



def create_model(config):
    """모델 생성 함수"""

    loss_logging_callback = LossLoggingCallback()

    # 학습률 스케줄러 설정
    if config.use_lr_scheduler:
        lr_scheduler_cls = config.lr_scheduler_cls
        lr_scheduler_kwargs = config.lr_scheduler_kwargs
    else:
        lr_scheduler_cls = None
        lr_scheduler_kwargs = None

    # NLinearModel 인스턴스 생성
    if config.model_name == "dlinear":
        model = DLinearModel(
            input_chunk_length=config.input_chunk_length,
            output_chunk_length=config.output_chunk_length,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            optimizer_kwargs={"lr": config.lr},
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            loss_fn=config.loss_fn,
            random_state=config.random_state,
            use_reversible_instance_norm=config.use_instance_norm,
            pl_trainer_kwargs={
                "callbacks": [config.early_stopping, config.checkpoint_callback, loss_logging_callback],
                "enable_checkpointing": True
            }
        )
        
    elif config.model_name =="nlinear":
        model = NLinearModel(
            input_chunk_length=config.input_chunk_length,
            output_chunk_length=config.output_chunk_length,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            optimizer_kwargs={"lr": config.lr},
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            loss_fn=config.loss_fn,
            random_state=config.random_state,
            use_reversible_instance_norm=config.use_instance_norm,
            pl_trainer_kwargs={
                "callbacks": [config.early_stopping, config.checkpoint_callback, loss_logging_callback],
                "enable_checkpointing": True
            }
        )
        
    elif config.model_name =="tide":
        
        model = TiDEModel(
            input_chunk_length=config.input_chunk_length,
            output_chunk_length=config.output_chunk_length,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            optimizer_kwargs={"lr": config.lr},
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            loss_fn=config.loss_fn,
            random_state=config.random_state,
            use_reversible_instance_norm=config.use_instance_norm,
            # TiDE 모델 관련 설정
            num_encoder_layers = config.num_encoder_layers,
            num_decoder_layers = config.num_decoder_layers,
            decoder_output_dim = config.decoder_output_dim,
            hidden_size = config.hidden_size,
            temporal_decoder_hidden = config.temporal_decoder_hidden,
            use_layer_norm = config.use_layer_norm,
            pl_trainer_kwargs={
                "callbacks": [config.early_stopping, config.checkpoint_callback, loss_logging_callback],
                "enable_checkpointing": True
            }
        )
        
    elif config.model_name == "tsmixer":
        model = TSMixerModel(
            input_chunk_length=config.input_chunk_length,
            output_chunk_length=config.output_chunk_length,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            optimizer_kwargs={"lr": config.lr},
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            loss_fn=config.loss_fn,
            random_state=config.random_state,
            num_blocks=config.num_block,
            use_reversible_instance_norm=config.use_instance_norm,
            normalize_before  = config.use_layer_norm, # 레이어 정규화 여부
            pl_trainer_kwargs={
                "callbacks": [config.early_stopping, config.checkpoint_callback, loss_logging_callback],
                "enable_checkpointing": True
            }
        )
        

    return model, loss_logging_callback


def __main__():
    return create_model(config)

if __name__ == "__main__":
    model = __main__()
    print(model)
