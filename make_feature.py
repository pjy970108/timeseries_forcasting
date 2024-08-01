import pandas as pd
from utils import csv_data, save_data
import config
import pandas as pd
from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.volume import (
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.trend import (
    MACD,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
    WMAIndicator,
)

from tqdm import tqdm 


def create_features(df_, target='close'):
    df = df_.copy()

    df['range'] = df['high'].shift(1) - df['low'].shift(1)
    df['volatility_breakout'] = df['open'] + df['range'] * 0.5

    windows = [5, 10, 20]
    for i in tqdm(windows, desc='Processing windows'): 
        df['pct_change_close_{}'.format(i)] = df['close'].pct_change(i)

        df['std_close_{}'.format(i)] = df['close'].rolling(i).std()

    df = _get_ta_features(df, windows=windows)
    
    df['median'] = (df['high'] + df['low']) / 2
    df.insert(0, 'target', df[target])
    
    if 'time' in df.columns:
        df.drop(columns=['time', target], inplace=True)
    else:
        df.drop(columns=[target], inplace=True)
    
    return df.dropna().reset_index(drop=True)


def _get_ta_features(df_:pd.DataFrame, fillna:bool=False, windows:list=[5, 10, 20]) -> pd.DataFrame:
    # 파생변수를 만드는 함수
    df = df_.copy()
    
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)

    # Momentum Indicators ---------------------------------------------------------------------------------------------------
    # Relative Strength Index (RSI)
    df['momentum_rsi'] = RSIIndicator(
        close=close, window=14, fillna=fillna
    ).rsi()
    
    # Stochastic RSI
    stoch_rsi_indicator = StochRSIIndicator(
        close=close, window=14, smooth1=3, smooth2=3, fillna=fillna
    )
    df['momentum_stoch_rsi'] = stoch_rsi_indicator.stochrsi()
    df['momentum_stoch_rsi_k'] = stoch_rsi_indicator.stochrsi_k()
    df['momentum_stoch_rsi_d'] = stoch_rsi_indicator.stochrsi_d()
    
    # True strength index (TSI)
    df['momentum_tsi'] = TSIIndicator(
        close=close, window_slow=25, window_fast=13, fillna=fillna
    ).tsi()

    # Ultimate Oscillator
    df['momentum_uo'] = UltimateOscillator(
        high=high, low=low, close=close, window1=7, window2=14, window3=28, weight1=4, weight2=2, weight3=1, fillna=fillna
    ).ultimate_oscillator()
    
    # Stochastic Oscillator
    stoch_oscillator = StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3, fillna=fillna
    )
    df['momentum_stoch'] = stoch_oscillator.stoch()
    df['momentum_stoch_signal'] = stoch_oscillator.stoch_signal()
    
    # Williams %R
    df['momentum_wr'] = WilliamsRIndicator(
        high=high, low=low, close=close, lbp=14, fillna=fillna
    ).williams_r()
    
    # Awesome Oscillator
    df['momentum_ao'] = AwesomeOscillatorIndicator(
        high=high, low=low, window1=5, window2=34, fillna=fillna
    ).awesome_oscillator()

    # Rate of Change (ROC)
    df['momentum_roc'] = ROCIndicator(
        close=close, window=12, fillna=fillna
    ).roc()

    # Percentage Price Oscillator (PPO)
    percentage_price_oscillator = PercentagePriceOscillator(
        close=close, window_slow=26, window_fast=12, window_sign=9, fillna=9
    )
    df['momentum_ppo'] = percentage_price_oscillator.ppo()
    df['momentum_ppo_signal'] = percentage_price_oscillator.ppo_signal()
    df['momentum_ppo_hist'] = percentage_price_oscillator.ppo_hist()

    # Percentage Volume Oscillator (PVO)
    percentage_volume_oscillator = PercentageVolumeOscillator(
        volume=volume, window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df['momentum_pvo'] = percentage_volume_oscillator.pvo()
    df['momentum_pvo_signal'] = percentage_volume_oscillator.pvo_signal()
    df['momentum_pvo_hist'] = percentage_volume_oscillator.pvo_hist()

    # Kaufman’s Adaptive Moving Average (KAMA)
    df['momentum_kama'] = KAMAIndicator(
        close=close, window=10, pow1=2, pow2=30, fillna=fillna
    ).kama()

    # On-balance volume (OBV)
    df['volume_obv'] = OnBalanceVolumeIndicator(
        close=close, volume=volume, fillna=fillna
    ).on_balance_volume()
    
    # Chaikin Money Flow (CMF)
    df['volume_cmf'] = ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, window=20, fillna=fillna
    ).chaikin_money_flow()
    
    # Force Index (FI)
    df['volume_fi'] = ForceIndexIndicator(
        close=close, volume=volume, window=13, fillna=fillna
    ).force_index()
    
    # Ease of movement (EoM, EMV)
    ease_of_movement_indicator = EaseOfMovementIndicator(
        high=high, low=low, volume=volume, window=14, fillna=fillna
    )
    df['volume_em_{}'.format(14)] = ease_of_movement_indicator.ease_of_movement()
    df['volume_sma_em_{}'.format(14)] = ease_of_movement_indicator.sma_ease_of_movement()

    # Volume-price trend (VPT)
    df['volume_vpt'] = VolumePriceTrendIndicator(
        close=close, volume=volume, fillna=fillna
    ).volume_price_trend()

    # Volume Weighted Average Price (VWAP)
    df['volume_vwap'] = VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume, window=14, fillna=fillna
    ).volume_weighted_average_price()
    
    # Money Flow Index (MFI)
    df['volume_mfi'] = MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14, fillna=fillna
    ).money_flow_index()
    
    # Negative Volume Index (NVI)
    df['volume_nvi'] = NegativeVolumeIndexIndicator(
        close=close, volume=volume, fillna=fillna
    ).negative_volume_index()
    
    # Volatility Indicators ---------------------------------------------------------------------------------------------------
    # Bollinger Bands
    bollinger_bands = BollingerBands(
        close=close, window=20, window_dev=2, fillna=fillna
    )
    df['volatility_bbh'] = bollinger_bands.bollinger_hband()
    df['volatility_bbhi'] = bollinger_bands.bollinger_hband_indicator()
    df['volatility_bbl'] = bollinger_bands.bollinger_lband()
    df['volatility_bbli'] = bollinger_bands.bollinger_lband_indicator()
    df['volatility_bbm'] = bollinger_bands.bollinger_mavg()
    df['volatility_bbp'] = bollinger_bands.bollinger_pband()
    df['volatility_bbw'] = bollinger_bands.bollinger_wband()
    
    # Donchian Channel
    donchian_channel = DonchianChannel(
        high=high, low=low, close=close, window=20, offset=0, fillna=0
    )
    df['volatility_dch'] = donchian_channel.donchian_channel_hband()
    df['volatility_dcl'] = donchian_channel.donchian_channel_lband()
    df['volatility_dcm'] = donchian_channel.donchian_channel_mband()
    df['volatility_dcp'] = donchian_channel.donchian_channel_pband()
    df['volatility_dcw'] = donchian_channel.donchian_channel_wband()
    
    # Average True Range (ATR)
    df['volatility_atr'] = AverageTrueRange(
        high=high, low=low, close=close, window=14, fillna=fillna
    ).average_true_range()
    
    # Ulcer Index
    df['volatility_ui'] = UlcerIndex(
        close=close, window=14, fillna=fillna
    ).ulcer_index(    )
    
    # Trend Indicators ---------------------------------------------------------------------------------------------------
    # Moving Average Convergence Divergence (MACD)
    macd = MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df['trend_macd'] = macd.macd()
    df['trend_macd_diff'] = macd.macd_diff()
    df['trend_macd_signal'] = macd.macd_signal()
    
    for i in windows:
        # Simple Moving Average (SMA)
        df['trend_sma_{}'.format(i)] = SMAIndicator(
            close=close, window=i, fillna=fillna
        ).sma_indicator()
    
        # Exponential Moving Average (EMA)
        df['trend_ema_{}'.format(i)] = EMAIndicator(
            close=close, window=i, fillna=fillna
        ).ema_indicator()
        
        # Weighted Moving Average (WMA)
        df['trend_wma_{}'.format(i)] = WMAIndicator(
            close=close, window=i, fillna=fillna
        ).wma()

    # Vortex Indicator (VI)
    vortex_indicator = VortexIndicator(
        high=high, low=low, close=close, window=14, fillna=fillna
    )
    df['trend_vortex_ind_diff'] = vortex_indicator.vortex_indicator_diff()
    df['trend_vortex_ind_pos'] = vortex_indicator.vortex_indicator_pos()
    df['trend_vortex_ind_neg'] = vortex_indicator.vortex_indicator_neg()
 
    # Tripple Exponential Smoothed Moving Average (TRIX)
    df['trend_trix'] = TRIXIndicator(
        close=close, window=15, fillna=fillna
    ).trix()
 
    # Mass Index (MI)
    df['trend_mass_index'] = MassIndex(
        high=high, low=low, window_fast=9, window_slow=25, fillna=fillna
    ).mass_index()
    
    # Detrended Price Oscillator (DPO)
    df['trend_dpo'] = DPOIndicator(
        close=close, window=20, fillna=fillna
    ).dpo()
    
    # KST Oscillator (KST Signal)
    kst_indicator = KSTIndicator(
        close=close, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9, fillna=fillna
    )
    df['trend_kst'] = kst_indicator.kst()
    df['trend_kst_diff'] = kst_indicator.kst_diff()
    df['trend_kst_sig'] = kst_indicator.kst_sig()
 
    # Ichimoku Kinkō Hyō (Ichimoku)
    ichimoku_indicator = IchimokuIndicator(
        high=high, low=low, window1=9, window2=26, window3=52, visual=False, fillna=False
    )
    df['trend_ichimoku_a'] = ichimoku_indicator.ichimoku_a()
    df['trend_ichimoku_b'] = ichimoku_indicator.ichimoku_b()
    df['trend_ichimoku_base'] = ichimoku_indicator.ichimoku_base_line()
    df['trend_ichimoku_conv'] = ichimoku_indicator.ichimoku_conversion_line()
 
    # Schaff Trend Cycle (STC)
    df['trend_stc'] = STCIndicator(
        close=close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=fillna
    ).stc()

    # Commodity Channel Index (CCI)
    df['trend_cci'] = CCIIndicator(
        high=high, low=low, close=close, window=20, constant=0.015, fillna=fillna
    ).cci()
 
    # Parabolic Stop and Reverse (Parabolic SAR)
    psar_indicator = PSARIndicator(
        high=high, low=low, close=close, step=0.02, max_step=0.2, fillna=fillna
    )
    df['trend_psar'] = psar_indicator.psar()
    df['trend_psar_up'] = psar_indicator.psar_up()
    df['trend_psar_up_indicator'] = psar_indicator.psar_up_indicator()
    df['trend_psar_down'] = psar_indicator.psar_down()
    df['trend_psar_down_indicator'] = psar_indicator.psar_down_indicator()
    df[['trend_psar_up', 'trend_psar_down']] = df[['trend_psar_up', 'trend_psar_down']].fillna(0) # psar 컬럼 null 채우기
    
    return df


def get_features(config):
    """특성이 포함된 데이터프레임을 반환하는 함수"""
    try:
        origin_data = csv_data(config.data_dir, config.origin_file)
        origin_data.drop_duplicates(inplace=True)
        origin_data.sort_values(by='open_time', inplace=True)
        origin_data.reset_index(drop=True, inplace=True)
    except FileNotFoundError:
        # 파일이 없으면 원본 데이터를 로드하고 특성을 생성
        print("FileNotFoundError")
        
    data = create_features(origin_data, target=config.target_column)
    save_data(data, config.data_dir, config.features_file)
    
    return data


def __main__():
    get_features(config)

if __name__ == "__main__":
    __main__()