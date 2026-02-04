#!/usr/bin/env python3
"""Script to fetch historical data from IBKR."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.fetcher import IBKRDataFetcher
from data.preprocessor import DataPreprocessor
from data.features import FeatureEngineer
from utils.logger import setup_logger

import pandas as pd

logger = setup_logger("fetch_data", level="INFO")


def fetch_continuous_all_at_once(fetcher, years: int, bar_size: str, use_rth: bool) -> pd.DataFrame:
    """
    Fetch multi-year data using continuous contract in one request.
    IBKR continuous contracts don't allow endDateTime, so we fetch everything at once.

    Args:
        fetcher: IBKRDataFetcher instance
        years: Number of years to fetch
        bar_size: Bar size string
        use_rth: Regular trading hours only

    Returns:
        DataFrame with historical data
    """
    from ib_insync import ContFuture, util

    contract = ContFuture(
        symbol=settings.SYMBOL,
        exchange=settings.EXCHANGE,
        currency=settings.CURRENCY,
    )

    logger.info(f"Fetching {years} years of data using CONTINUOUS contract...")
    logger.info(f"This may take a few minutes. Please wait...")

    try:
        bars = fetcher.ib.reqHistoricalData(
            contract=contract,
            endDateTime="",  # Must be empty for ContFuture
            durationStr=f"{years} Y",
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=1,
            timeout=600,  # 10 minute timeout for large requests
        )

        if bars:
            df = util.df(bars)
            logger.info(f"Got {len(df)} bars")
            return df
        else:
            logger.warning("No data returned from continuous contract")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching continuous data: {e}")
        return pd.DataFrame()


def fetch_by_contract_expiries(fetcher, years: int, bar_size: str, use_rth: bool, chunk_days: int = 5) -> pd.DataFrame:
    """
    Fetch historical data using available contracts from IBKR.
    First discovers what contracts are available, then fetches from each.

    Args:
        fetcher: IBKRDataFetcher instance
        years: Number of years to fetch (used as a guide)
        bar_size: Bar size string
        use_rth: Regular trading hours only
        chunk_days: Days per chunk within each contract

    Returns:
        Combined DataFrame
    """
    from ib_insync import Future, util

    all_data = []

    # First, discover what contracts are actually available
    logger.info("Discovering available MNQ contracts...")

    base_contract = Future(
        symbol=settings.SYMBOL,
        exchange=settings.EXCHANGE,
        currency=settings.CURRENCY,
    )

    details = fetcher.ib.reqContractDetails(base_contract)

    if not details:
        logger.error("No contracts found!")
        return pd.DataFrame()

    # Sort by expiry and get available contracts
    available_contracts = sorted(
        [d.contract for d in details],
        key=lambda c: c.lastTradeDateOrContractMonth
    )

    logger.info(f"Found {len(available_contracts)} available contracts")

    # Fetch from each available contract
    for i, contract in enumerate(available_contracts):
        expiry = contract.lastTradeDateOrContractMonth
        logger.info(f"[{i+1}/{len(available_contracts)}] Fetching {contract.localSymbol} ({expiry})...")

        try:
            # Fetch maximum available data for this contract
            # Try different durations, starting with largest
            for duration in ["1 Y", "6 M", "3 M", "1 M"]:
                try:
                    bars = fetcher.ib.reqHistoricalData(
                        contract=contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=use_rth,
                        formatDate=1,
                        timeout=120,
                    )

                    if bars:
                        df_chunk = util.df(bars)
                        all_data.append(df_chunk)
                        logger.info(f"  Got {len(df_chunk)} bars ({duration}): {df_chunk['date'].min()} to {df_chunk['date'].max()}")
                        break

                except Exception as e:
                    if "pacing" in str(e).lower():
                        logger.warning(f"  Pacing violation, waiting...")
                        time.sleep(10)
                    continue

            # Pace requests
            time.sleep(3)

        except Exception as e:
            logger.warning(f"  Error fetching {expiry}: {e}")
            time.sleep(5)

    if not all_data:
        return pd.DataFrame()

    # Combine all contracts
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date'])
    combined = combined.sort_values('date')

    logger.info(f"Total: {len(combined)} bars from {len(all_data)} contracts")
    if len(combined) > 0:
        logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

    return combined


def fetch_in_chunks(fetcher, total_days: int, bar_size: str, use_rth: bool, chunk_days: int = 5) -> pd.DataFrame:
    """
    Fetch historical data in chunks using front-month contract.
    Only works for shorter durations (< 60 days typically).

    Args:
        fetcher: IBKRDataFetcher instance
        total_days: Total days of data to fetch
        bar_size: Bar size string
        use_rth: Regular trading hours only
        chunk_days: Days per chunk (default 5 for 5-min bars)

    Returns:
        Combined DataFrame
    """
    from ib_insync import util

    all_data = []
    end_date = datetime.now()
    days_fetched = 0
    total_chunks = (total_days + chunk_days - 1) // chunk_days
    chunk_num = 0
    failed_chunks = 0
    max_consecutive_fails = 10

    contract = fetcher._contract
    logger.info(f"Using front-month contract for {total_days} days")
    logger.info(f"Fetching in {chunk_days}-day chunks (~{total_chunks} requests)...")

    consecutive_fails = 0

    while days_fetched < total_days:
        chunk_num += 1
        progress = days_fetched / total_days * 100

        try:
            print(f"\r[{chunk_num}/{total_chunks}] {progress:.1f}% - Fetching {end_date.strftime('%Y-%m-%d')}...", end="", flush=True)

            bars = fetcher.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_date,
                durationStr=f"{chunk_days} D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
                timeout=60,
            )

            if bars:
                df_chunk = util.df(bars)
                all_data.append(df_chunk)
                print(f"\r[{chunk_num}/{total_chunks}] {progress:.1f}% - {end_date.strftime('%Y-%m-%d')}: {len(df_chunk)} bars", flush=True)
                consecutive_fails = 0
            else:
                print(f"\r[{chunk_num}/{total_chunks}] {progress:.1f}% - {end_date.strftime('%Y-%m-%d')}: No data", flush=True)
                consecutive_fails += 1

            end_date = end_date - timedelta(days=chunk_days)
            days_fetched += chunk_days
            time.sleep(2.5)

        except Exception as e:
            failed_chunks += 1
            consecutive_fails += 1
            print(f"\r[{chunk_num}/{total_chunks}] {progress:.1f}% - {end_date.strftime('%Y-%m-%d')}: Error", flush=True)
            end_date = end_date - timedelta(days=chunk_days)
            days_fetched += chunk_days
            time.sleep(5)

        if consecutive_fails >= max_consecutive_fails:
            logger.warning(f"\n{max_consecutive_fails} consecutive failures - stopping")
            break

    print()

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date'])
    combined = combined.sort_values('date')

    logger.info(f"Total: {len(combined)} bars fetched")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Fetch historical data from IBKR")
    parser.add_argument(
        "--duration",
        type=str,
        default="1 Y",
        help="Duration of data to fetch (e.g., '3 Y', '1 Y', '6 M', '30 D')",
    )
    parser.add_argument(
        "--bar-size",
        type=str,
        default="5 mins",
        help="Bar size (e.g., '5 mins', '1 hour', '1 day')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--rth-only",
        action="store_true",
        help="Fetch regular trading hours only",
    )
    parser.add_argument(
        "--compute-features",
        action="store_true",
        help="Compute technical indicators",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading account",
    )
    parser.add_argument(
        "--update",
        type=str,
        default=None,
        help="Update existing data file",
    )
    parser.add_argument(
        "--expiry",
        type=str,
        default=None,
        help="Contract expiry in YYYYMM format (e.g., '202603'). Default: front month",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Use continuous contract (deprecated, use chunked fetching instead)",
    )

    args = parser.parse_args()

    logger.info("Starting data fetch...")
    logger.info(f"Duration: {args.duration}, Bar size: {args.bar_size}")

    # Parse duration to days
    duration_str = args.duration.upper()
    if "Y" in duration_str:
        total_days = int(duration_str.replace("Y", "").strip()) * 365
    elif "M" in duration_str:
        total_days = int(duration_str.replace("M", "").strip()) * 30
    elif "D" in duration_str:
        total_days = int(duration_str.replace("D", "").strip())
    else:
        total_days = 365  # Default 1 year

    try:
        with IBKRDataFetcher(paper=args.paper) as fetcher:
            if args.update:
                # Update existing file
                fetcher.get_mnq_contract(expiry=args.expiry)
                logger.info(f"Updating existing file: {args.update}")
                df = fetcher.update_data(args.update)
            elif total_days > 90:
                # Multi-year data - use continuous contract with single request
                years = max(1, total_days // 365)
                logger.info(f"Fetching {years} years of data...")

                df = fetch_continuous_all_at_once(
                    fetcher=fetcher,
                    years=years,
                    bar_size=args.bar_size,
                    use_rth=args.rth_only,
                )

                # Process the raw data
                if not df.empty:
                    df = fetcher._process_bar_data(df)

            elif total_days > 30:
                # Medium duration - use chunked fetching with front-month
                fetcher.get_mnq_contract(expiry=args.expiry)
                logger.info(f"Using chunked fetching for {total_days} days...")
                df = fetch_in_chunks(
                    fetcher=fetcher,
                    total_days=total_days,
                    bar_size=args.bar_size,
                    use_rth=args.rth_only,
                    chunk_days=5,
                )
                if not df.empty:
                    df = fetcher._process_bar_data(df)
            else:
                # Standard fetch for short durations
                fetcher.get_mnq_contract(expiry=args.expiry)
                logger.info("Fetching historical data...")
                df = fetcher.fetch_historical_data(
                    duration=args.duration,
                    bar_size=args.bar_size,
                    use_rth=args.rth_only,
                )

            if df.empty:
                logger.error("No data received")
                return 1

            logger.info(f"Received {len(df)} bars")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

            # Preprocess data
            logger.info("Preprocessing data...")
            preprocessor = DataPreprocessor()
            df = preprocessor.clean_data(df)

            if args.rth_only:
                df = preprocessor.filter_market_hours(df, rth_only=True)

            # Compute features if requested
            if args.compute_features:
                logger.info("Computing technical indicators...")
                feature_engineer = FeatureEngineer()
                df = feature_engineer.compute_all_features(df)
                logger.info(f"Computed {len(feature_engineer.feature_names)} features")

            # Save to parquet
            output_file = args.output
            if output_file is None:
                start = df.index[0].strftime("%Y%m%d")
                end = df.index[-1].strftime("%Y%m%d")
                suffix = "_features" if args.compute_features else ""
                output_file = f"{settings.SYMBOL}_{start}_{end}{suffix}.parquet"

            filepath = fetcher.save_to_parquet(df, output_file)
            logger.info(f"Data saved to {filepath}")

            # Print summary
            print("\n" + "=" * 50)
            print("DATA FETCH SUMMARY")
            print("=" * 50)
            print(f"Symbol: {settings.SYMBOL}")
            print(f"Total bars: {len(df)}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Columns: {list(df.columns)}")
            print(f"File: {filepath}")
            print("=" * 50 + "\n")

            return 0

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        logger.error("Make sure TWS or IB Gateway is running")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
