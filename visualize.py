import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from config import config
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis, plot_hist, stationarity_test_kpss, stationarity_test_adf
from darts.metrics.metrics import mse, r2_score, mape, mae, rmse

class LossLoggingCallback(Callback):
    def __init__(self, plot_file=None):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        config.update_model_name()
        self.plot_file = f'{config.model_name}_loss_plot.png'

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(f'{config.model_dir}/'+ self.plot_file)
        
        
def visualize_pred(real_data, pred_data, config):
    # 플롯 크기 설정
    plt.figure(figsize=(12, 6))

    # 타겟 샘플과 예측값 플롯
    real_data.plot(label="True", color="blue")
    pred_data.plot(label="Prediction", color="red")

    # 제목과 레이블 추가
    plt.title("True vs Prediction")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # 범례 추가
    plt.legend(loc="upper right")

    # 그리드 추가 (선택 사항)
    plt.grid(True)

    # 플롯 표시
    plt.show()
    plt.savefig(f'{config.model_dir}/prediction_plot.png')
    
    
def check_stationarity(data, max_lag, bins):
    col_name = data.components[0]
    print(f"{col_name} seasonality: ", check_seasonality(data, max_lag))
    plot_acf(data, max_lag)
    plot_hist(data, bins)
    _, kpss_p_value, _, _ = stationarity_test_kpss(data)
    
    # 귀무가설 : 데이터가 정상성을 띤다
    # 대립가설 : 데이터가 비정상적이다
    # 귀무가설 기각(데이터가 비정상적)
    if kpss_p_value < 0.05:
        print(f"KPSS test: {kpss_p_value}, {col_name} is not stationary")
    # 귀무가설 채택(데이터가 정상적)
    else:
        print(f"KPSS test: {kpss_p_value}, {col_name} is stationary")
    
    _, adf_p_value, _, _, _, _ = stationarity_test_adf(data)
    # 귀무가설 : 데이터가 단위근을 가지고 비정상적
    # 대립가설 : 데이터가 단위근을 가지지 않고 정상적
    # 귀무가설 기각(시계열이 정상)
    if adf_p_value < 0.05:
        print(f"ADF test: {adf_p_value}, {col_name} is stationary")
    # 귀무가설 채택 (시계열이 비정상)
    else:
        print(f"ADF test: {adf_p_value}, {col_name} is not stationary")
    


def evaluate_metrics(real_data, pred_data):
    evaluate_dict = {"mse": mse(real_data, pred_data),
                     "r2_score": r2_score(real_data, pred_data),
                     "mape": mape(real_data, pred_data),
                     "mae": mae(real_data, pred_data),
                     "rmse": rmse(real_data, pred_data)}
    # MSE 계산
    print(f"MSE: {mse(real_data, pred_data)}")

    # R2 계산
    print(f"R2 Score: {r2_score(real_data, pred_data)}")

    # MAPE 계산
    print(f"MAPE: {mape(real_data, pred_data)}")

    # MAE 계산
    print(f"MAE: {mae(real_data, pred_data)}")

    # RMSE 계산
    print(f"RMSE: {rmse(real_data, pred_data)}")
    
    return evaluate_dict
    
    