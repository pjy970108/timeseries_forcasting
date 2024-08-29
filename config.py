import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

class Config:
    def __init__(self):
        # 기본 경로 설정
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'dataset')
        self.model_dir = os.path.join(self.base_dir, 'models')

        # 데이터 관련 설정
        self.origin_file = 'BTC_1m.csv'
        self.target_column = 'median'
        self.features_file = f'{self.target_column}_features.pkl'
        self.cov_cols = ['open_time', 'trend_macd', 'trend_macd_diff', 'trend_macd_signal', 'trend_sma_5', 'trend_ema_5', 'trend_wma_5', 'trend_sma_20', 'trend_ema_20', 'trend_wma_20', 'trend_vortex_ind_diff', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_diff', 'trend_kst_sig', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_ichimoku_base', 'trend_ichimoku_conv', 'trend_stc', 'trend_cci']

        # 데이터 전처리 관련 설정
        self.freq = '1min'
        self.time_col = 'open_time'
        self.scaler = 'standard'
        self.save_data = True
        self.fill_params = {'interpolation_method': 'linear', 'order': 3}
        self.val_split = 0.2
        self.test_split = 0.2

        # 모델 관련 설정
        self.model_name = "tsmixer" # nlinear, tide, tsmixer
        self.input_chunk_length = 20
        self.output_chunk_length = 10
        self.n_epochs = 10
        self.batch_size = 64
        self.lr = 0.001
        self.use_lr_scheduler = True
        self.lr_scheduler_cls = torch.optim.lr_scheduler.StepLR
        self.lr_scheduler_kwargs = {"step_size": 10, "gamma": 0.1}

        self.random_state = 42 

        self.using_cov = True
        self.update_model_name()
        

        self.work_dir = 'models'
        self.force_reset = False
        self.use_instance_norm = False
        self.loss_fn = torch.nn.MSELoss()

        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=0.0001,
            mode="min"
        )

        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="../darts/checkpoints",
            filename="best_{self.model_file_name}_checkpoint",
            save_top_k=1,
            mode="min",
        )
        
        ## TIDE 모델 관련 설정
        self.num_encoder_layers = 1 # Dense encoder layer 설정 -> 많을수록 더 복잡한 패턴 학습
        self.num_decoder_layers  = 1 # Dense decoder layer 설정 -> 많을수록 복잔한 패턴 학습
        self.decoder_output_dim = 4 # Decoder output 차원 설정 -> 높을수록 많은 정보가 포함됨
        self.hidden_size = 256 # encoder, decoder hidden size 설정 높을수록 많은 정보 학습
        self.temporal_decoder_hidden = 32 # 시간적 decoder의 폭, 더 클수록 복잡한 패턴 학습
        self.use_layer_norm = False # 잔차 블록에서 정규화 사용여부
        
        ## TSMIZER 관련 설정
        self.num_block = 2
        
        

        self.mode = "train"
        self.eval_pred = True

    def update_model_name(self):
        self.model_file_name = f'{self.model_name}_cov' if self.using_cov else f'{self.model_name}_no_cov'
        self.model_file = f'{self.model_file_name}.pt'

config = Config()



