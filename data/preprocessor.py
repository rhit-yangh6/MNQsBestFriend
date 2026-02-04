"""Data preprocessing module for cleaning and normalizing data."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime, time
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses OHLCV data for model training."""

    def __init__(
        self,
        scaler_type: str = "robust",
        fill_method: str = "ffill",
    ):
        """
        Initialize the preprocessor.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            fill_method: Method for filling missing values
        """
        self.scaler_type = scaler_type
        self.fill_method = fill_method
        self.scaler: Optional[StandardScaler] = None
        self.fitted = False
        self._feature_columns: List[str] = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw OHLCV data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep="first")]
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate rows")

        # Sort by index
        df = df.sort_index()

        # Handle missing values
        df = self._handle_missing_values(df)

        # Remove invalid OHLC relationships
        df = self._validate_ohlc(df)

        # Remove outliers
        df = self._remove_outliers(df)

        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        df = df.copy()

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")

        # Fill missing values
        if self.fill_method == "ffill":
            df = df.ffill()
        elif self.fill_method == "bfill":
            df = df.bfill()
        elif self.fill_method == "interpolate":
            df = df.interpolate(method="time")

        # Drop any remaining NaN rows
        df = df.dropna()

        return df

    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships."""
        df = df.copy()

        # High should be >= Open, Close, Low
        invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])

        # Low should be <= Open, Close, High
        invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])

        invalid_mask = invalid_high | invalid_low
        if invalid_mask.any():
            logger.warning(f"Removing {invalid_mask.sum()} rows with invalid OHLC relationships")
            df = df[~invalid_mask]

        return df

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        column: str = "close",
        z_threshold: float = 10.0,
    ) -> pd.DataFrame:
        """Remove statistical outliers."""
        df = df.copy()

        # Calculate returns for outlier detection
        returns = df[column].pct_change()
        z_scores = (returns - returns.mean()) / returns.std()

        outliers = np.abs(z_scores) > z_threshold
        if outliers.any():
            logger.warning(f"Removing {outliers.sum()} outlier rows")
            df = df[~outliers]

        return df

    def filter_market_hours(
        self,
        df: pd.DataFrame,
        rth_only: bool = False,
    ) -> pd.DataFrame:
        """
        Filter data by market hours.

        Args:
            df: DataFrame with datetime index
            rth_only: If True, only keep regular trading hours

        Returns:
            Filtered DataFrame
        """
        df = df.copy()

        if rth_only:
            # Regular Trading Hours: 9:30 AM - 4:00 PM ET
            rth_start = time(settings.RTH_START[0], settings.RTH_START[1])
            rth_end = time(settings.RTH_END[0], settings.RTH_END[1])

            mask = (df.index.time >= rth_start) & (df.index.time < rth_end)
            df = df[mask]

            # Also filter weekends
            df = df[df.index.dayofweek < 5]

            logger.info(f"Filtered to RTH only: {len(df)} rows")

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: DataFrame with datetime index

        Returns:
            DataFrame with time features
        """
        df = df.copy()

        # Hour of day (cyclical encoding)
        hour = df.index.hour + df.index.minute / 60
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of week (cyclical encoding)
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Is regular trading hours
        rth_start = time(settings.RTH_START[0], settings.RTH_START[1])
        rth_end = time(settings.RTH_END[0], settings.RTH_END[1])
        df["is_rth"] = ((df.index.time >= rth_start) & (df.index.time < rth_end)).astype(int)

        # Time to market close (in hours)
        minutes_to_close = (
            settings.RTH_END[0] * 60 + settings.RTH_END[1]
            - df.index.hour * 60
            - df.index.minute
        )
        df["time_to_close"] = minutes_to_close / 60  # Convert to hours
        df["time_to_close"] = df["time_to_close"].clip(lower=0)

        return df

    def create_scaler(self) -> None:
        """Create the appropriate scaler."""
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> "DataPreprocessor":
        """
        Fit the scaler on training data.

        Args:
            df: Training DataFrame
            feature_columns: Columns to scale (None = all numeric)

        Returns:
            Self for chaining
        """
        if self.scaler is None:
            self.create_scaler()

        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self._feature_columns = feature_columns
        self.scaler.fit(df[feature_columns])
        self.fitted = True

        logger.info(f"Fitted scaler on {len(feature_columns)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        df = df.copy()
        scaled_values = self.scaler.transform(df[self._feature_columns])
        df[self._feature_columns] = scaled_values

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform
            feature_columns: Columns to scale

        Returns:
            Transformed DataFrame
        """
        self.fit(df, feature_columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data.

        Args:
            df: Scaled DataFrame

        Returns:
            Original scale DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        df = df.copy()
        original_values = self.scaler.inverse_transform(df[self._feature_columns])
        df[self._feature_columns] = original_values

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets chronologically.

        Args:
            df: Full DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(
            f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 50,
        target_column: str = "close",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.

        Args:
            df: DataFrame with features
            sequence_length: Length of each sequence
            target_column: Column to predict

        Returns:
            Tuple of (X sequences, y targets)
        """
        feature_cols = [col for col in df.columns if col != target_column]
        features = df[feature_cols].values
        targets = df[target_column].values

        X, y = [], []
        for i in range(sequence_length, len(df)):
            X.append(features[i - sequence_length : i])
            y.append(targets[i])

        return np.array(X), np.array(y)

    def save_scaler(self, filepath: Path) -> None:
        """Save the fitted scaler to disk."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        joblib.dump(
            {
                "scaler": self.scaler,
                "feature_columns": self._feature_columns,
                "scaler_type": self.scaler_type,
            },
            filepath,
        )
        logger.info(f"Saved scaler to {filepath}")

    def load_scaler(self, filepath: Path) -> None:
        """Load a previously saved scaler."""
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self._feature_columns = data["feature_columns"]
        self.scaler_type = data["scaler_type"]
        self.fitted = True
        logger.info(f"Loaded scaler from {filepath}")
