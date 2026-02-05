# MNQ Futures Trading AI

A Reinforcement Learning-based trading system for MNQ (Micro E-mini Nasdaq-100) futures on 5-minute timeframe using Interactive Brokers API.

## Features

- **Data Pipeline**: Automated IBKR data fetching with technical indicator computation
- **RL Environment**: Custom Gymnasium environment with realistic trading simulation
- **Model Training**: PPO/A2C algorithms with LSTM/Attention feature extractors
- **Backtesting**: Event-driven backtesting with comprehensive performance metrics
- **Paper Trading**: Real-time paper trading with IBKR integration
- **Live Trading**: Production-ready live trading with risk management

## Project Structure

```
MNQModel/
├── config/
│   ├── settings.py          # Global configuration
│   └── ib_config.py         # Interactive Brokers credentials
├── data/
│   ├── fetcher.py           # IBKR data fetching
│   ├── preprocessor.py      # Data cleaning & normalization
│   └── features.py          # Technical indicator computation
├── env/
│   ├── trading_env.py       # Custom Gym environment
│   ├── rewards.py           # Reward function definitions
│   └── position_sizer.py    # Position sizing logic
├── models/
│   ├── agent.py             # RL agent wrapper
│   ├── networks.py          # Neural network architectures
│   └── callbacks.py         # Training callbacks
├── backtest/
│   ├── engine.py            # Backtesting simulation
│   └── metrics.py           # Performance metrics
├── trading/
│   ├── paper_trader.py      # Paper trading execution
│   ├── live_trader.py       # Live trading execution
│   └── order_manager.py     # Order handling
├── utils/
│   ├── logger.py            # Logging utilities
│   └── visualize.py         # Charts & dashboards
├── scripts/
│   ├── fetch_data.py        # Download historical data
│   ├── train.py             # Model training script
│   ├── backtest.py          # Run backtests
│   └── trade.py             # Paper/live trading
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MNQModel
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install TA-Lib for additional indicators:
```bash
# macOS
brew install ta-lib
pip install ta-lib

# Ubuntu
sudo apt-get install libta-lib-dev
pip install ta-lib
```

## Configuration

### Interactive Brokers Setup

1. Install and configure TWS or IB Gateway
2. Enable API connections in TWS/Gateway settings
3. Set environment variables (optional):
```bash
export IB_HOST=127.0.0.1
export IB_PAPER_PORT=7497
export IB_LIVE_PORT=7496
export IB_CLIENT_ID=1
```

## Usage

### 1. Fetch Historical Data

```bash
python scripts/fetch_data.py --duration "1 Y" --bar-size "5 mins" --compute-features
```

Options:
- `--duration`: Data duration (e.g., "1 Y", "6 M", "30 D")
- `--bar-size`: Bar size (e.g., "5 mins", "1 hour")
- `--rth-only`: Regular trading hours only
- `--compute-features`: Compute technical indicators
- `--update`: Update existing data file

### 2. Train Model

```bash
python scripts/train.py \
    --data data/storage/MNQ_20230101_20240101.parquet \
    --timesteps 500000 \
    --algorithm PPO \
    --feature-extractor lstm \
    --tensorboard
```

Options:
- `--algorithm`: PPO or A2C
- `--feature-extractor`: lstm, gru, attention
- `--fixed-sl-ticks`: Stop loss in ticks (default 200)
- `--walk-forward`: Use walk-forward validation
- `--tensorboard`: Enable TensorBoard logging

Note: Uses sparse reward system (rewards only on trade completion).

### 3. Backtest

```bash
python scripts/backtest.py \
    --model models/saved/train_20240101_120000 \
    --data data/storage/test_data.parquet \
    --initial-capital 10000
```

### 4. Paper Trading

```bash
python scripts/trade.py \
    --model models/saved/train_20240101_120000 \
    --mode paper \
    --max-position 1 \
    --max-daily-loss 500
```

### 5. Live Trading

⚠️ **WARNING: Live trading involves real money. Use at your own risk.**

```bash
python scripts/trade.py \
    --model models/saved/train_20240101_120000 \
    --mode live \
    --max-position 1 \
    --max-daily-loss 500 \
    --use-stop-loss
```

## Model Architecture

### Feature Extractor

The system supports two feature extractors:

1. **LSTM Extractor**: Processes sequential market data through LSTM layers
   - Captures temporal dependencies
   - Suitable for trend-following strategies

2. **Attention Extractor**: Uses self-attention mechanism
   - Identifies important time steps
   - Better for pattern recognition

### State Space

```python
state = {
    'market_features': [...],  # Technical indicators (lookback_window x n_features)
    'position_info': [
        position,          # Current position (-1, 0, 1)
        unrealized_pnl,    # Normalized unrealized P&L
        time_in_position,  # Bars since entry
        balance_ratio,     # Current equity / initial equity
    ],
}
```

### Action Space

- `0`: HOLD - Maintain current position
- `1`: BUY - Open/hold long position
- `2`: SELL - Open/hold short position
- `3`: CLOSE - Close current position

## Risk Management

### Position Sizing Methods

1. **Fixed**: Constant position size
2. **Fixed Fractional**: Risk a fixed percentage of account
3. **Kelly Criterion**: Optimal position sizing based on win rate
4. **Volatility-based**: Adjust size based on ATR

### Risk Controls

- Maximum position size limit
- Maximum daily loss limit
- Automatic stop-loss orders
- Slippage modeling

## Performance Metrics

The system calculates comprehensive metrics:

- **Return Metrics**: Total return, annualized return, CAGR
- **Risk Metrics**: Volatility, VaR, CVaR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Metrics**: Win rate, profit factor, expectancy
- **Drawdown**: Max drawdown, average drawdown, recovery factor

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black .
isort .
flake8 .
```

### Type Checking

```bash
mypy .
```

## API Reference

### TradingEnv

```python
from env.trading_env import TradingEnv

env = TradingEnv(
    df=data,
    feature_columns=['rsi', 'macd', ...],
    initial_balance=10000.0,
    lookback_window=50,
    stop_loss_ticks=200.0,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

### TradingAgent

```python
from models.agent import TradingAgent

agent = TradingAgent(
    env=env,
    algorithm='PPO',
    feature_extractor='lstm',
    learning_rate=3e-4,
)

agent.train(total_timesteps=500000)
agent.save('model_path')

action, _ = agent.predict(observation)
```

### BacktestEngine

```python
from backtest.engine import BacktestEngine

engine = BacktestEngine(
    df=data,
    initial_capital=10000.0,
    commission=0.62,
)

results = engine.run_with_model(model, feature_columns)
```

## Troubleshooting

### Common Issues

1. **IBKR Connection Failed**
   - Ensure TWS/Gateway is running
   - Check API settings are enabled
   - Verify port numbers

2. **Out of Memory During Training**
   - Reduce batch size
   - Reduce lookback window
   - Use gradient accumulation

3. **Model Not Learning**
   - Check data normalization
   - Reduce learning rate
   - Increase exploration (entropy coefficient)

## Disclaimer

This software is for educational and research purposes only. Trading futures involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use this software at your own risk.

## License

MIT License - See LICENSE file for details.
