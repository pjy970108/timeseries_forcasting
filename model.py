import config
from darts.models import NLinearModel
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
        pl_trainer_kwargs={
            "callbacks": [config.early_stopping, config.checkpoint_callback, loss_logging_callback],
            "enable_checkpointing": True
        },
    )

    return model, loss_logging_callback


def __main__():
    return create_model(config)

if __name__ == "__main__":
    model = __main__()
    print(model)
