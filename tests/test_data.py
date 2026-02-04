"""Tests for data modules."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import DataPreprocessor
from data.features import FeatureEngineer


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    n_bars = 200
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="5T")

    np.random.seed(42)
    close = 15000 + np.cumsum(np.random.randn(n_bars) * 5)

    df = pd.DataFrame({
        "open": close + np.random.randn(n_bars) * 2,
        "high": close + abs(np.random.randn(n_bars) * 3),
        "low": close - abs(np.random.randn(n_bars) * 3),
        "close": close,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=dates)

    return df


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_clean_data(self, sample_ohlcv):
        """Test data cleaning."""
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(sample_ohlcv)

        assert len(cleaned) > 0
        assert not cleaned.isnull().any().any()

    def test_clean_data_removes_duplicates(self, sample_ohlcv):
        """Test duplicate removal."""
        # Add duplicate rows
        df_with_dups = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:5]])

        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(df_with_dups)

        assert len(cleaned) == len(sample_ohlcv)

    def test_handle_missing_values(self, sample_ohlcv):
        """Test missing value handling."""
        # Introduce missing values
        df = sample_ohlcv.copy()
        df.loc[df.index[10:15], "close"] = np.nan

        preprocessor = DataPreprocessor(fill_method="ffill")
        cleaned = preprocessor.clean_data(df)

        assert not cleaned["close"].isnull().any()

    def test_validate_ohlc(self, sample_ohlcv):
        """Test OHLC validation."""
        # Introduce invalid OHLC (high < low)
        df = sample_ohlcv.copy()
        df.loc[df.index[5], "high"] = df.loc[df.index[5], "low"] - 10

        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(df)

        # Invalid row should be removed
        assert len(cleaned) < len(df)

    def test_add_time_features(self, sample_ohlcv):
        """Test time feature addition."""
        preprocessor = DataPreprocessor()
        df = preprocessor.add_time_features(sample_ohlcv)

        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "dow_sin" in df.columns
        assert "dow_cos" in df.columns
        assert "is_rth" in df.columns

    def test_scaler_fit_transform(self, sample_ohlcv):
        """Test scaler fitting and transformation."""
        preprocessor = DataPreprocessor(scaler_type="standard")

        # Fit and transform
        scaled = preprocessor.fit_transform(
            sample_ohlcv,
            feature_columns=["close", "volume"],
        )

        assert preprocessor.fitted
        # Scaled data should have mean ~0, std ~1
        assert abs(scaled["close"].mean()) < 1
        assert abs(scaled["close"].std() - 1) < 0.5

    def test_scaler_inverse_transform(self, sample_ohlcv):
        """Test inverse transformation."""
        preprocessor = DataPreprocessor(scaler_type="standard")

        original_close = sample_ohlcv["close"].copy()

        # Fit and transform
        scaled = preprocessor.fit_transform(
            sample_ohlcv,
            feature_columns=["close"],
        )

        # Inverse transform
        recovered = preprocessor.inverse_transform(scaled)

        # Should recover original values
        np.testing.assert_array_almost_equal(
            recovered["close"].values,
            original_close.values,
            decimal=5,
        )

    def test_scaler_save_load(self, sample_ohlcv):
        """Test scaler save and load."""
        preprocessor = DataPreprocessor(scaler_type="robust")
        preprocessor.fit_transform(sample_ohlcv, feature_columns=["close"])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "scaler.joblib"

            # Save
            preprocessor.save_scaler(filepath)
            assert filepath.exists()

            # Load into new instance
            new_preprocessor = DataPreprocessor()
            new_preprocessor.load_scaler(filepath)

            assert new_preprocessor.fitted
            assert new_preprocessor._feature_columns == preprocessor._feature_columns

    def test_split_data(self, sample_ohlcv):
        """Test data splitting."""
        preprocessor = DataPreprocessor()

        train, val, test = preprocessor.split_data(
            sample_ohlcv,
            train_ratio=0.7,
            val_ratio=0.15,
        )

        # Check sizes
        total = len(sample_ohlcv)
        assert len(train) == int(total * 0.7)
        assert len(train) + len(val) + len(test) == total

        # Check chronological order
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_compute_all_features(self, sample_ohlcv):
        """Test computing all features."""
        engineer = FeatureEngineer()
        df = engineer.compute_all_features(sample_ohlcv)

        # Should have many features
        assert len(df.columns) > len(sample_ohlcv.columns)
        assert len(engineer.feature_names) > 0

        # Check no NaN in output
        assert not df.isnull().any().any()

    def test_add_price_features(self, sample_ohlcv):
        """Test price-based features."""
        engineer = FeatureEngineer()
        df = engineer.add_price_features(sample_ohlcv)

        assert "returns" in df.columns
        assert "log_returns" in df.columns
        assert "hl_range" in df.columns

    def test_add_momentum_indicators(self, sample_ohlcv):
        """Test momentum indicators."""
        engineer = FeatureEngineer()
        df = engineer.add_momentum_indicators(sample_ohlcv)

        assert "rsi" in df.columns
        assert "macd" in df.columns
        assert "stoch_k" in df.columns

        # RSI should be between 0 and 100
        assert df["rsi"].dropna().min() >= 0
        assert df["rsi"].dropna().max() <= 100

    def test_add_trend_indicators(self, sample_ohlcv):
        """Test trend indicators."""
        engineer = FeatureEngineer()
        df = engineer.add_trend_indicators(sample_ohlcv)

        assert "ema_9" in df.columns
        assert "ema_21" in df.columns
        assert "adx" in df.columns

    def test_add_volatility_indicators(self, sample_ohlcv):
        """Test volatility indicators."""
        engineer = FeatureEngineer()
        df = engineer.add_volatility_indicators(sample_ohlcv)

        assert "atr" in df.columns
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns

        # BB upper should be above lower
        valid = df.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_add_volume_indicators(self, sample_ohlcv):
        """Test volume indicators."""
        engineer = FeatureEngineer()
        df = engineer.add_volume_indicators(sample_ohlcv)

        assert "volume_ma_10" in df.columns
        assert "obv" in df.columns

    def test_add_custom_features(self, sample_ohlcv):
        """Test custom features."""
        # First add required dependencies
        engineer = FeatureEngineer()
        df = engineer.add_trend_indicators(sample_ohlcv)
        df = engineer.add_custom_features(df)

        assert "bullish_bar" in df.columns
        assert "higher_high" in df.columns
        assert "zscore_20" in df.columns

    def test_add_time_features(self, sample_ohlcv):
        """Test time features."""
        engineer = FeatureEngineer()
        df = engineer.add_time_features(sample_ohlcv)

        assert "hour_sin" in df.columns
        assert "is_morning" in df.columns

    def test_get_feature_names(self, sample_ohlcv):
        """Test feature name retrieval."""
        engineer = FeatureEngineer()
        engineer.compute_all_features(sample_ohlcv)

        names = engineer.get_feature_names()
        assert len(names) > 0
        assert "returns" in names

    def test_select_features(self, sample_ohlcv):
        """Test feature selection."""
        engineer = FeatureEngineer()
        df = engineer.compute_all_features(sample_ohlcv)

        selected = engineer.select_features(df, ["returns", "rsi"])
        assert list(selected.columns) == ["returns", "rsi"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
