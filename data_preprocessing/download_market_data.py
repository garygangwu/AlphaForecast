from dotenv import load_dotenv
import os
import requests
import pandas as pd
from pathlib import Path
from io import StringIO

# Load environment variables
load_dotenv(override=True)
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Define data directory
DATA_DIR = Path("../market_data")
DATA_DIR.mkdir(exist_ok=True)


def fetch_vix_data_from_cboe():
    """
    Fetch VIX data directly from CBOE CDN.

    Args:
        start_date (str): Start date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame with VIX data
    """
    # CBOE VIX data URL
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    print(f"Fetching VIX data from CBOE CDN...")
    response = requests.get(url)

    if response.status_code == 200:
        # Read CSV data
        df = pd.read_csv(StringIO(response.text))
        print(f"  Successfully fetched {len(df)} rows from CBOE")

        # Rename columns to match our format
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close']

        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        return df
    else:
        print(f"  Error fetching VIX data from CBOE: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def fetch_and_save_ohlcv(symbol, key, save_path):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={key}&datatype=csv&outputsize=full"
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"[{symbol}] Saved to {save_path}")

def main():
    print("="*60)
    print("MARKET DATA DOWNLOAD")
    print("="*60)
    print(f"Output directory: {DATA_DIR}")
    print("="*60)

    print("  Source: CBOE CDN (https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv)")
    vix_df = fetch_vix_data_from_cboe()
    if vix_df is not None:
        vix_file = DATA_DIR / "VIX.csv"
        vix_df.to_csv(vix_file, index=False)
        print(f"  VIX data saved to: {vix_file}")
        print(f"  Date range: {vix_df['Date'].min()} to {vix_df['Date'].max()}")
        print(f"  Total records: {len(vix_df)}")
    else:
        print("  Failed to download VIX data")

    print("  Source: Alpha Vantage API (SPY, QQQ, IWM, IWM, DIA)")
    fetch_and_save_ohlcv("SPY", ALPHA_VANTAGE_API_KEY, DATA_DIR / "SPY_raw.csv")
    fetch_and_save_ohlcv("QQQ", ALPHA_VANTAGE_API_KEY, DATA_DIR / "QQQ_raw.csv")
    fetch_and_save_ohlcv("IWM", ALPHA_VANTAGE_API_KEY, DATA_DIR / "IWM_raw.csv")
    fetch_and_save_ohlcv("DIA", ALPHA_VANTAGE_API_KEY, DATA_DIR / "DIA_raw.csv")

    print("\nProcessing raw market data files...")

    # Process each ETF file
    for etf in ['SPY', 'QQQ', 'IWM', 'DIA']:
        raw_file = DATA_DIR / f"{etf}_raw.csv"
        output_file = DATA_DIR / f"{etf}.csv"

        if raw_file.exists():
            # Read raw data
            df = pd.read_csv(raw_file)

            # Select and rename columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Convert Date to datetime and sort
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            # Save processed file
            df.to_csv(output_file, index=False)
            print(f"  Processed {etf}: {len(df):,} rows ({df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d})")

            # Remove raw file
            raw_file.unlink()
        else:
            print(f"  Warning: Raw file not found for {etf}")

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print("Files downloaded:")
    for file in DATA_DIR.glob("*.csv"):
        print(f"  - {file.name}")

    print(f"\nAll market data saved to: {DATA_DIR}")
    print("Note: These datasets can be used as additional features for market sentiment analysis.")

if __name__ == "__main__":
    main()
