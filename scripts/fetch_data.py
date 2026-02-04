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


def fetch_in_chunks(fetcher, total_days: int, bar_size: str, use_rth: bool, chunk_days: int = 5) -> pd.DataFrame:
    """
    Fetch historical data in chunks to avoid IBKR limitations.

    Args:
        fetcher: IBKRDataFetcher instance
        total_days: Total days of data to fetch
        bar_size: Bar size string
        use_rth: Regular trading hours only
        chunk_days: Days per chunk (default 5 for 5-min bars)

    Returns:
        Combined DataFrame
    """
    all_data = []
    end_date = datetime.now()
    days_fetched = 0

    logger.info(f"Fetching {total_days} days of data in {chunk_days}-day chunks...")

    while days_fetched < total_days:
        try:
            logger.info(f"Fetching chunk ending {end_date.strftime('%Y-%m-%d')} ({days_fetched}/{total_days} days done)")

            bars = fetcher.ib.reqHistoricalData(
                contract=fetcher._contract,
                endDateTime=end_date,
                durationStr=f"{chunk_days} D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
            )

            if bars:
                from ib_insync import util
                df_chunk = util.df(bars)
                all_data.append(df_chunk)
                logger.info(f"  Got {len(df_chunk)} bars")
            else:
                logger.warning(f"  No data for this chunk")

            # Move to next chunk
            end_date = end_date - timedelta(days=chunk_days)
            days_fetched += chunk_days

            # Pace the requests to avoid IBKR limits
            time.sleep(2)

        except Exception as e:
            logger.warning(f"  Error fetching chunk: {e}")
            # Move on even if one chunk fails
            end_date = end_date - timedelta(days=chunk_days)
            days_fetched += chunk_days
            time.sleep(5)

    if not all_data:
        return pd.DataFrame()

    # Combine all chunks
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
                # Use multi-year fetching for longer durations (iterate through contracts)
                years = max(1, total_days // 365)
                logger.info(f"Using multi-year fetching for {years} years of data...")
                df = fetcher.fetch_multi_year_data(
                    years=years,
                    bar_size=args.bar_size,
                    use_rth=args.rth_only,
                )
            elif total_days > 30:
                # Use chunked fetching for medium durations
                fetcher.get_mnq_contract(expiry=args.expiry)
                logger.info(f"Using chunked fetching for {total_days} days...")
                df = fetch_in_chunks(
                    fetcher=fetcher,
                    total_days=total_days,
                    bar_size=args.bar_size,
                    use_rth=args.rth_only,
                    chunk_days=5,  # 5 days per chunk for 5-min bars
                )
                # Process the raw data
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
