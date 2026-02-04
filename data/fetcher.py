"""IBKR data fetching module using ib_insync."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from ib_insync import IB, Future, ContFuture, util, Contract

from config.settings import settings
from config.ib_config import ib_config

logger = logging.getLogger(__name__)


class IBKRDataFetcher:
    """Fetches historical and real-time data from Interactive Brokers."""

    def __init__(
        self,
        paper: bool = True,
        use_gateway: bool = False,
        client_id: Optional[int] = None,
    ):
        """
        Initialize the IBKR data fetcher.

        Args:
            paper: Use paper trading account
            use_gateway: Use IB Gateway instead of TWS
            client_id: Override default client ID
        """
        self.ib = IB()
        self.paper = paper
        self.use_gateway = use_gateway
        self.client_id = client_id or ib_config.CLIENT_ID
        self.connected = False
        self._contract: Optional[Contract] = None

    def connect(self) -> bool:
        """
        Connect to TWS/Gateway.

        Returns:
            True if connection successful
        """
        try:
            port = ib_config.get_port(self.paper, self.use_gateway)
            self.ib.connect(
                host=ib_config.HOST,
                port=port,
                clientId=self.client_id,
                timeout=ib_config.TIMEOUT,
            )
            self.connected = True
            logger.info(f"Connected to IBKR on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def get_mnq_contract(self, expiry: Optional[str] = None) -> Future:
        """
        Get the MNQ futures contract.

        Args:
            expiry: Contract expiry in YYYYMM format, or None for front month

        Returns:
            Future contract object
        """
        if expiry:
            contract = Future(
                symbol=settings.SYMBOL,
                exchange=settings.EXCHANGE,
                currency=settings.CURRENCY,
                lastTradeDateOrContractMonth=expiry,
            )
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                self._contract = qualified[0]
                logger.info(f"Qualified contract: {self._contract}")
                return self._contract
            else:
                raise ValueError(f"Could not qualify contract for {settings.SYMBOL} with expiry {expiry}")
        else:
            # Get front month contract by requesting contract details
            contract = Future(
                symbol=settings.SYMBOL,
                exchange=settings.EXCHANGE,
                currency=settings.CURRENCY,
            )

            # Request all available contracts
            details = self.ib.reqContractDetails(contract)

            if not details:
                raise ValueError(f"No contracts found for {settings.SYMBOL}")

            # Sort by expiry date and select the front month (earliest expiry)
            details_sorted = sorted(
                details,
                key=lambda d: d.contract.lastTradeDateOrContractMonth
            )

            # Select the front month contract
            front_month = details_sorted[0].contract

            # Qualify it
            qualified = self.ib.qualifyContracts(front_month)
            if qualified:
                self._contract = qualified[0]
                logger.info(f"Selected front month contract: {self._contract.localSymbol} (expires {self._contract.lastTradeDateOrContractMonth})")
                return self._contract
            else:
                raise ValueError(f"Could not qualify front month contract for {settings.SYMBOL}")

    def get_continuous_contract(self) -> ContFuture:
        """
        Get continuous futures contract for historical data.
        This allows fetching multi-year historical data across contract rollovers.

        Returns:
            Continuous future contract
        """
        # Create continuous contract - don't qualify it
        # IBKR handles continuous contracts specially for historical data
        self._contract = ContFuture(
            symbol=settings.SYMBOL,
            exchange=settings.EXCHANGE,
            currency=settings.CURRENCY,
        )
        logger.info(f"Using continuous contract: {settings.SYMBOL} on {settings.EXCHANGE}")
        return self._contract

    def fetch_historical_data(
        self,
        duration: str = "5 Y",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            duration: Duration string (e.g., "5 Y", "1 M", "5 D")
            bar_size: Bar size string (e.g., "5 mins", "1 hour", "1 day")
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)
            use_rth: Use regular trading hours only
            end_date: End date for historical data (not used for continuous contracts)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")

        if self._contract is None:
            self.get_mnq_contract()

        logger.info(
            f"Fetching {duration} of {bar_size} data for {self._contract.symbol}"
        )

        # Check if using continuous contract (ContFuture)
        is_continuous = self._contract.secType == "CONTFUT"

        if is_continuous:
            # Continuous contracts don't allow end date
            bars = self.ib.reqHistoricalData(
                contract=self._contract,
                endDateTime="",  # Empty string = now
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )
        else:
            end_dt = end_date or datetime.now()
            bars = self.ib.reqHistoricalData(
                contract=self._contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

        if not bars:
            logger.warning("No historical data returned")
            return pd.DataFrame()

        df = util.df(bars)
        df = self._process_bar_data(df)

        logger.info(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        return df

    def fetch_historical_data_chunked(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        bar_size: str = "5 mins",
        chunk_days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch historical data in chunks to avoid IBKR limits.

        Args:
            start_date: Start date for data
            end_date: End date for data
            bar_size: Bar size string
            chunk_days: Days per chunk

        Returns:
            Combined DataFrame
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")

        end_date = end_date or datetime.now()
        all_data = []

        current_end = end_date
        while current_end > start_date:
            chunk_start = max(current_end - timedelta(days=chunk_days), start_date)
            duration_days = (current_end - chunk_start).days + 1

            try:
                df = self.fetch_historical_data(
                    duration=f"{duration_days} D",
                    bar_size=bar_size,
                    end_date=current_end,
                )

                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Fetched chunk ending {current_end.date()}")

                # Avoid pacing violations
                self.ib.sleep(2)

            except Exception as e:
                logger.error(f"Error fetching chunk ending {current_end}: {e}")

            current_end = chunk_start - timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        return combined

    def _process_bar_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw bar data into standard format.

        Args:
            df: Raw DataFrame from ib_insync

        Returns:
            Processed DataFrame
        """
        df = df.copy()

        # Rename columns to standard names
        column_map = {
            "date": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "average": "vwap",
            "barCount": "bar_count",
        }

        df = df.rename(columns=column_map)

        # Set timestamp as index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        # Ensure numeric types
        numeric_cols = ["open", "high", "low", "close", "volume", "vwap", "bar_count"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def save_to_parquet(
        self, df: pd.DataFrame, filename: Optional[str] = None
    ) -> Path:
        """
        Save DataFrame to Parquet format.

        Args:
            df: DataFrame to save
            filename: Optional filename, default uses symbol and date range

        Returns:
            Path to saved file
        """
        if filename is None:
            start = df.index[0].strftime("%Y%m%d")
            end = df.index[-1].strftime("%Y%m%d")
            filename = f"{settings.SYMBOL}_{start}_{end}.parquet"

        filepath = settings.DATA_DIR / filename
        df.to_parquet(filepath, engine="pyarrow", compression="snappy")
        logger.info(f"Saved data to {filepath}")
        return filepath

    def load_from_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from Parquet file.

        Args:
            filename: Filename to load

        Returns:
            Loaded DataFrame
        """
        filepath = settings.DATA_DIR / filename
        df = pd.read_parquet(filepath, engine="pyarrow")
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df

    def update_data(self, existing_file: str) -> pd.DataFrame:
        """
        Update existing data file with new data.

        Args:
            existing_file: Path to existing parquet file

        Returns:
            Updated DataFrame
        """
        existing_df = self.load_from_parquet(existing_file)
        last_date = existing_df.index[-1]

        # Fetch new data from last date to now
        new_df = self.fetch_historical_data(
            duration=f"{(datetime.now() - last_date.to_pydatetime()).days + 1} D",
            end_date=datetime.now(),
        )

        if new_df.empty:
            logger.info("No new data available")
            return existing_df

        # Combine and remove duplicates
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        # Save updated data
        self.save_to_parquet(combined, existing_file)

        logger.info(f"Updated data with {len(new_df)} new bars")
        return combined

    def stream_realtime_bars(self, callback) -> None:
        """
        Stream real-time 5-second bars.

        Args:
            callback: Function to call with each new bar
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")

        if self._contract is None:
            self.get_mnq_contract()

        bars = self.ib.reqRealTimeBars(
            contract=self._contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=False,
        )

        bars.updateEvent += callback
        logger.info(f"Started streaming real-time bars for {self._contract.symbol}")

    def get_contract_details(self) -> dict:
        """
        Get contract details for the current contract.

        Returns:
            Dictionary with contract details
        """
        if self._contract is None:
            self.get_mnq_contract()

        details = self.ib.reqContractDetails(self._contract)
        if details:
            detail = details[0]
            return {
                "symbol": detail.contract.symbol,
                "expiry": detail.contract.lastTradeDateOrContractMonth,
                "multiplier": detail.contract.multiplier,
                "min_tick": detail.minTick,
                "trading_hours": detail.tradingHours,
                "liquid_hours": detail.liquidHours,
            }
        return {}

    def get_historical_expiries(self, years_back: int = 3) -> List[str]:
        """
        Generate list of quarterly expiry dates going back N years.
        MNQ contracts expire in March (H), June (M), September (U), December (Z).

        Args:
            years_back: Number of years to go back

        Returns:
            List of expiry dates in YYYYMM format, ordered oldest to newest
        """
        import time
        now = datetime.now()
        expiries = []

        # Quarterly months: March=3, June=6, September=9, December=12
        quarter_months = [3, 6, 9, 12]

        # Go back N years
        start_year = now.year - years_back
        end_year = now.year

        for year in range(start_year, end_year + 1):
            for month in quarter_months:
                expiry_date = datetime(year, month, 1)
                # Only include if the expiry is in the past or current quarter
                if expiry_date <= now:
                    expiries.append(f"{year}{month:02d}")

        # Add next quarter if we're close to it
        current_month = now.month
        for month in quarter_months:
            if month > current_month:
                expiries.append(f"{now.year}{month:02d}")
                break
        else:
            # Current month is past December, add next year's March
            expiries.append(f"{now.year + 1}03")

        return sorted(expiries)

    def fetch_multi_year_data(
        self,
        years: int = 3,
        bar_size: str = "5 mins",
        use_rth: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch multiple years of historical data using continuous contract.
        Fetches year by year to show progress and avoid timeouts.

        Args:
            years: Number of years of data to fetch
            bar_size: Bar size string
            use_rth: Use regular trading hours only

        Returns:
            Combined DataFrame with all historical data
        """
        import time
        import threading
        import sys

        if not self.connected:
            raise ConnectionError("Not connected to IBKR")

        # Use continuous contract for historical data
        self._contract = ContFuture(
            symbol=settings.SYMBOL,
            exchange=settings.EXCHANGE,
            currency=settings.CURRENCY,
        )

        print(f"\n{'='*50}")
        print(f"Fetching {years} years of {settings.SYMBOL} 5-min data")
        print(f"Using continuous contract (ContFuture)")
        print(f"{'='*50}\n")

        # Progress indicator
        stop_spinner = threading.Event()

        def show_progress():
            start_time = time.time()
            spinner = ['|', '/', '-', '\\']
            i = 0
            while not stop_spinner.is_set():
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                sys.stdout.write(f'\r  Waiting for IBKR response... {spinner[i % 4]} [{mins:02d}:{secs:02d}]')
                sys.stdout.flush()
                i += 1
                time.sleep(0.25)
            sys.stdout.write('\r' + ' ' * 60 + '\r')
            sys.stdout.flush()

        all_data = []

        # Try fetching year by year
        for year_num in range(years, 0, -1):
            print(f"[{years - year_num + 1}/{years}] Requesting {year_num} year(s) of data...")

            # Start progress spinner in background
            stop_spinner.clear()
            spinner_thread = threading.Thread(target=show_progress)
            spinner_thread.start()

            try:
                bars = self.ib.reqHistoricalData(
                    contract=self._contract,
                    endDateTime="",  # Must be empty for ContFuture
                    durationStr=f"{year_num} Y",
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=1,
                    timeout=600,  # 10 minute timeout
                )

                stop_spinner.set()
                spinner_thread.join()

                if bars:
                    df = util.df(bars)
                    df = self._process_bar_data(df)
                    print(f"  ✓ Got {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")
                    return df  # Return immediately on success
                else:
                    print(f"  ✗ No data returned for {year_num} year(s)")

            except Exception as e:
                stop_spinner.set()
                spinner_thread.join()
                print(f"  ✗ Error: {e}")

            # Wait before trying shorter duration
            time.sleep(2)

        print("\nFailed to fetch data with continuous contract.")
        print("Falling back to front-month contract...")

        # Fallback: use front month and get whatever is available
        self.get_mnq_contract()
        try:
            bars = self.ib.reqHistoricalData(
                contract=self._contract,
                endDateTime="",
                durationStr="60 D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
            )

            if bars:
                df = util.df(bars)
                df = self._process_bar_data(df)
                print(f"  Got {len(df)} bars from front-month contract")
                return df
        except Exception as e:
            print(f"  Fallback also failed: {e}")

        return pd.DataFrame()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
