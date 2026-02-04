"""Feature engineering module for technical indicators."""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

from config.settings import settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes technical indicators and features for trading."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names: List[str] = []

    def compute_all_features(
        self,
        df: pd.DataFrame,
        include_time_features: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all technical indicators and features.

        Args:
            df: DataFrame with OHLCV data
            include_time_features: Whether to include time-based features

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Price-based features
        df = self.add_price_features(df)

        # Technical indicators
        df = self.add_momentum_indicators(df)
        df = self.add_trend_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)

        # Custom features
        df = self.add_custom_features(df)

        # Time features
        if include_time_features:
            df = self.add_time_features(df)

        # Drop NaN rows from indicator calculations
        df = df.dropna()

        # Store feature names (excluding OHLCV)
        base_cols = ["open", "high", "low", "close", "volume", "vwap", "bar_count"]
        self.feature_names = [col for col in df.columns if col not in base_cols]

        logger.info(f"Computed {len(self.feature_names)} features")
        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Price momentum at different windows
        for window in settings.FEATURE_WINDOWS:
            df[f"momentum_{window}"] = df["close"].pct_change(window)
            df[f"roc_{window}"] = (df["close"] - df["close"].shift(window)) / df["close"].shift(window) * 100

        # High-Low range
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["oc_range"] = (df["close"] - df["open"]) / df["open"]

        # Price position within bar
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

        # Gap
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df = df.copy()

        # RSI
        rsi = RSIIndicator(close=df["close"], window=settings.RSI_PERIOD)
        df["rsi"] = rsi.rsi()
        df["rsi_normalized"] = (df["rsi"] - 50) / 50  # Normalize to [-1, 1]

        # RSI at different periods
        for period in [7, 21]:
            rsi_temp = RSIIndicator(close=df["close"], window=period)
            df[f"rsi_{period}"] = rsi_temp.rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
            smooth_window=3,
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # MACD
        macd = MACD(
            close=df["close"],
            window_slow=settings.MACD_SLOW,
            window_fast=settings.MACD_FAST,
            window_sign=settings.MACD_SIGNAL,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # Normalize MACD by price
        df["macd_normalized"] = df["macd"] / df["close"] * 100
        df["macd_hist_normalized"] = df["macd_histogram"] / df["close"] * 100

        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        df = df.copy()

        # EMAs
        for period in settings.EMA_PERIODS:
            ema = EMAIndicator(close=df["close"], window=period)
            df[f"ema_{period}"] = ema.ema_indicator()
            # Distance from EMA (normalized)
            df[f"ema_{period}_dist"] = (df["close"] - df[f"ema_{period}"]) / df[f"ema_{period}"] * 100

        # EMA crossovers
        df["ema_9_21_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)
        df["ema_21_50_cross"] = (df["ema_21"] > df["ema_50"]).astype(int)

        # ADX (trend strength)
        adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # Linear regression slope
        for window in [10, 20]:
            df[f"linreg_slope_{window}"] = self._calculate_linreg_slope(df["close"], window)

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        df = df.copy()

        # ATR
        atr = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=settings.ATR_PERIOD,
        )
        df["atr"] = atr.average_true_range()
        df["atr_normalized"] = df["atr"] / df["close"] * 100  # ATR as % of price

        # Bollinger Bands
        bb = BollingerBands(
            close=df["close"],
            window=settings.BB_PERIOD,
            window_dev=settings.BB_STD,
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_position"] = bb.bollinger_pband()  # Position within bands [0, 1]

        # Rolling volatility
        for window in [10, 20, 50]:
            df[f"volatility_{window}"] = df["returns"].rolling(window).std() * np.sqrt(252 * 78)  # Annualized

        # Garman-Klass volatility
        df["gk_volatility"] = self._garman_klass_volatility(df, window=20)

        # Parkinson volatility (high-low based)
        df["parkinson_vol"] = self._parkinson_volatility(df, window=20)

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        df = df.copy()

        # Volume moving averages
        for window in [10, 20, 50]:
            df[f"volume_ma_{window}"] = df["volume"].rolling(window).mean()
            df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_ma_{window}"]

        # VWAP distance
        if "vwap" in df.columns:
            df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
        else:
            # Calculate VWAP
            vwap = VolumeWeightedAveragePrice(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
            )
            df["vwap"] = vwap.volume_weighted_average_price()
            df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        df["obv"] = obv.on_balance_volume()
        df["obv_normalized"] = (df["obv"] - df["obv"].rolling(20).mean()) / (
            df["obv"].rolling(20).std() + 1e-8
        )

        # Volume-price trend
        df["volume_price_trend"] = (df["returns"] * df["volume"]).rolling(10).sum()

        return df

    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom trading features."""
        df = df.copy()

        # Price action patterns
        df["doji"] = (abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8) < 0.1).astype(int)
        df["bullish_bar"] = (df["close"] > df["open"]).astype(int)

        # Higher highs / lower lows
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
        df["higher_high_count"] = df["higher_high"].rolling(5).sum()
        df["lower_low_count"] = df["lower_low"].rolling(5).sum()

        # Support/Resistance proximity
        df["recent_high"] = df["high"].rolling(20).max()
        df["recent_low"] = df["low"].rolling(20).min()
        df["resistance_dist"] = (df["recent_high"] - df["close"]) / df["close"] * 100
        df["support_dist"] = (df["close"] - df["recent_low"]) / df["close"] * 100

        # Mean reversion signals
        df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / (
            df["close"].rolling(20).std() + 1e-8
        )
        df["zscore_50"] = (df["close"] - df["close"].rolling(50).mean()) / (
            df["close"].rolling(50).std() + 1e-8
        )

        # Trend strength
        df["trend_strength"] = abs(df["ema_9_dist"]) + abs(df["ema_21_dist"])

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        # Hour of day (cyclical)
        hour = df.index.hour + df.index.minute / 60
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of week (cyclical)
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Session indicators
        df["is_morning"] = ((df.index.hour >= 9) & (df.index.hour < 12)).astype(int)
        df["is_afternoon"] = ((df.index.hour >= 12) & (df.index.hour < 16)).astype(int)
        df["is_overnight"] = ((df.index.hour < 9) | (df.index.hour >= 16)).astype(int)

        # First/last hour of RTH
        df["first_hour_rth"] = (
            (df.index.hour == 9) | ((df.index.hour == 10) & (df.index.minute < 30))
        ).astype(int)
        df["last_hour_rth"] = ((df.index.hour == 15)).astype(int)

        return df

    def _calculate_linreg_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate linear regression slope over rolling window."""
        def slope(arr):
            if len(arr) < window:
                return np.nan
            x = np.arange(len(arr))
            return np.polyfit(x, arr, 1)[0]

        return series.rolling(window).apply(slope, raw=True)

    def _garman_klass_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2

        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(gk.rolling(window).mean() * 252 * 78)

    def _parkinson_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        log_hl_sq = np.log(df["high"] / df["low"]) ** 2
        return np.sqrt(log_hl_sq.rolling(window).mean() / (4 * np.log(2)) * 252 * 78)

    def get_feature_names(self) -> List[str]:
        """Get list of computed feature names."""
        return self.feature_names

    def select_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Select specific features from DataFrame.

        Args:
            df: DataFrame with all features
            features: List of feature names to select (None = all)

        Returns:
            DataFrame with selected features
        """
        if features is None:
            features = self.feature_names

        return df[features]
