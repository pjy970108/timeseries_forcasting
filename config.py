import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch


# 기본 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'dataset')
model_dir = os.path.join(base_dir, 'models')


# 데이터 관련 설정, make_feature.py
origin_file = 'BTC_1m.csv'
target_column = 'median'
features_file = f'{target_column}_features.pkl'
cov_cols = ['open_time', 'trend_macd', 'trend_macd_diff', 'trend_macd_signal', 'trend_sma_5', 'trend_ema_5', 'trend_wma_5', 'trend_sma_20', 'trend_ema_20', 'trend_wma_20', 'trend_vortex_ind_diff', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_diff', 'trend_kst_sig', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_ichimoku_base', 'trend_ichimoku_conv', 'trend_stc', 'trend_cci']#, 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind']


# 데이터 전처리 관련 설정, make_dataset.py
freq = '1min'
time_col = 'open_time'
scaler = 'standard'
save_data = True
fill_params = {'interpolation_method': 'linear', # polynomial, spline, nearest, time, cubic, quadratic, pad, bfill
                'order': 3}
val_split = 0.2
test_split = 0.2



# 모델 관련 설정
input_chunk_length = 20 # 입력 데이터의 길이, 일자
output_chunk_length = 10 # 출력 데이터의 길이, 일자
n_epochs = 10 # 학습 횟수
batch_size = 64 # 배치사이즈
lr = 0.001 # 학습률
use_lr_scheduler = False # 스케쥴러 사용 여부
lr_scheduler_cls = torch.optim.lr_scheduler.StepLR # 스케쥴러
lr_scheduler_kwargs = {"step_size": 10, "gamma": 0.1} # 스케쥴러 설정

random_state = 42 
model_name = 'dliner_no_cov' # 불러올 모델 이름
work_dir = 'models' # 모델 경로
force_reset = False # 기존 모델에 덮어쓸지말지, True로 설정하면 이전에 저장된 모델이 있더라도 초기화
model_file = f'{model_name}.pt' # 저장될 모델 이름
use_instance_norm = False # 인스턴스 정규화 사용 여부
loss_fn = torch.nn.MSELoss() # 손실함수


early_stopping = EarlyStopping(
    monitor="val_loss", # 검증데이터 손실 최소화
    patience=5, # 개선되지 않는 에포크 
    min_delta=0.0001, # 개선되었는지 판단하는 최소 변화량 
    mode="min" # 손실이 최소화 되어야 하므로 최소화로 설정
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", # 검증데이터 최소화
    dirpath="../darts/checkpoints", # 저장될 경로
    filename=f"best_{model_name}_checkpoint", # 저장될 파일 이름
    save_top_k=1, #저장될 모델 개수
    mode="min",
)

using_cov = True # 공변량 사용 여부


## predict
mode = "train" # train, predict
eval_pred = True # 예측값 평가 여부



