from typing import Optional, Tuple

import torch
import torch.nn as nn

from darts.logging import raise_if
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on both ends of time series if kernel_size is odd
        if self.kernel_size % 2 == 1:
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
        # padding only at the front if kernel_size is even
        else:
            front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
            x = torch.cat([front, x], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.kernel_size = configs.kernel_size
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.use_RevIN = configs.use_RevIN
        self.rnn_type = configs.rnn_type if hasattr(configs, 'rnn_type') else None
        self.hidden_size = configs.hidden_size if hasattr(configs, 'hidden_size') else None
        self.num_layers = configs.num_layers if hasattr(configs, 'num_layers') else None

        # Check if the rnn_type is correct
        if self.rnn_type not in ['lstm', 'gru']:
            raise ValueError(f'Invalid rnn_type! Expected "lstm", "gru" but got {self.rnn_type}')

        # Decompsition Kernel Size
        self.decomposition = SeriesDecomp(self.kernel_size)
        
        # RevIN
        if self.use_RevIN:
            self.revin = RevIN(self.channels)
        
        # Initialize RNN and Linear layers
        self._initialize_layers()
            
    def _initialize_layers(self):
        # RNN layers
        if self.rnn_type:
            if self.rnn_type == 'lstm':
                RNNLayer = nn.LSTM
            elif self.rnn_type == 'gru':
                RNNLayer = nn.GRU

            if self.individual:
                self.RNN_Seasonal = nn.ModuleList([RNNLayer(1, 1, self.num_layers, batch_first=True) for _ in range(self.channels)])
                self.RNN_Trend = nn.ModuleList([RNNLayer(1, 1, self.num_layers, batch_first=True) for _ in range(self.channels)])
            else:
                self.RNN_Seasonal = RNNLayer(self.channels, self.hidden_size, self.num_layers, batch_first=True)
                self.RNN_Trend = RNNLayer(self.channels, self.hidden_size, self.num_layers, batch_first=True)
            
        # Linear layers
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.Linear_Trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
    def forward(self, x):
        # 1. Optional Instance Normalization
        if self.use_RevIN:
            x = self.revin(x, 'norm')
            
        # 2. Decomposition
        seasonal, trend = self.decomposition(x)
        
        # 3. RNN processing
        if self.rnn_type:
            if self.individual:
                for i in range(self.channels):
                    rnn_seasonal_output, _ = self.RNN_Seasonal[i](seasonal[:, :, i].unsqueeze(2))
                    rnn_trend_output, _ = self.RNN_Trend[i](trend[:, :, i].unsqueeze(2))
                    seasonal[:, :, i] = rnn_seasonal_output.squeeze(2)
                    trend[:, :, i] = rnn_trend_output.squeeze(2)
            else:
                seasonal, _ = self.RNN_Seasonal(seasonal)
                trend, _ = self.RNN_Trend(trend)
        
        # 4. Linear processing for Direct Multi Step (DMS) prediction: [Batch, Channel, Output length]
        seasonal, trend = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal.size(0), seasonal.size(1), self.pred_len], dtype=seasonal.dtype).to(seasonal.device)
            trend_output = torch.zeros([trend.size(0), trend.size(1), self.pred_len], dtype=trend.dtype).to(trend.device)            
            
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal)
            trend_output = self.Linear_Trend(trend)

        # 5. Sum the results
        x = seasonal_output + trend_output
        
        # 6. Reorder dimensions: [Batch, Output length, Channel]
        x = x.permute(0, 2, 1)
        
        # 7. Optional Instance Denormalization
        if self.use_RevIN:
            x = self.revin(x, 'denorm')
            
        return x