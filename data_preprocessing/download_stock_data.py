from dotenv import load_dotenv
import os
import requests
import time
import argparse

tickers = ['BX', 'CHD', 'WY', 'GFS', 'COF', 'META', 'PEG', 'RMD', 'WAB', 'NWSA', 'FIS', 'CZR', 'BAC', 'JKHY', 'LVS', 'BBY', 'CTAS', 'DIS', 'MO', 'PAYX', 'AMZN', 'MDT', 'MAR', 'CMG', 'MRVL', 'APH', 'HLT', 'TER', 'AVB', 'CB', 'SHOP', 'BXP', 'COR', 'TMO', 'F', 'CTSH', 'PFG', 'CAT', 'FRT', 'HBAN', 'SW', 'VRSN', 'MAS', 'JBHT', 'HON', 'HOLX', 'TTWO', 'LH', 'SWKS', 'GEV', 'KIM', 'VRSK', 'HPQ', 'NDAQ', 'LW', 'DHR', 'EXE', 'DTE', 'REGN', 'J', 'HUBB', 'SHW', 'CDW', 'MET', 'ORLY', 'NEM', 'MGM', 'GOOGL', 'CNP', 'HUM', 'MCD', 'ACGL', 'EG', 'DOV', 'EVRG', 'AZN', 'ABBV', 'EXR', 'IEX', 'EXPE', 'IDXX', 'SJM', 'CSX', 'TECH', 'AIG', 'IRM', 'CAG', 'PCG', 'TJX', 'ROP', 'LDOS', 'DOW', 'TFC', 'POOL', 'HES', 'SWK', 'HST', 'HSIC', 'WBA', 'HII', 'LKQ', 'AOS', 'RVTY', 'LUV', 'TEL', 'ADM', 'DELL', 'HIG', 'SOLV', 'PYPL', 'KEY', 'RCL', 'SO', 'LOW', 'PCAR', 'MTB', 'UAL', 'ED', 'SMCI', 'EXC', 'PSA', 'MCK', 'NTRS', 'LNT', 'CMI', 'KHC', 'FSLR', 'CF', 'MRNA', 'INTU', 'CL', 'FITB', 'CVS', 'FI', 'STLD', 'AON', 'PNR', 'DGX', 'TXN', 'ETR', 'NSC', 'LLY', 'AAPL', 'TSCO', 'XEL', 'BRO', 'ANSS', 'MSTR', 'FOX', 'LULU', 'DPZ', 'KMB', 'AWK', 'PTC', 'TRGP', 'CTVA', 'PPG', 'STT', 'AFL', 'AKAM', 'GPN', 'DAL', 'MDLZ', 'REG', 'DVA', 'NI', 'IFF', 'DRI', 'PFE', 'LMT', 'ARE', 'AVGO', 'BF-B', 'EL', 'TROW', 'ALLE', 'RSG', 'NDSN', 'COO', 'DECK', 'T', 'EXPD', 'EQIX', 'ULTA', 'MSCI', 'ON', 'APD', 'EQR', 'TGT', 'CPRT', 'CVX', 'TEAM', 'AMD', 'FTV', 'WFC', 'BG', 'URI', 'KR', 'AES', 'CHRW', 'MOH', 'WST', 'EA', 'BALL', 'VLO', 'AME', 'ADSK', 'PNW', 'CARR', 'AXP', 'PWR', 'JPM', 'PSX', 'FTNT', 'CSGP', 'MNST', 'USB', 'MPC', 'CAH', 'CPAY', 'ALGN', 'DLTR', 'NEE', 'PNC', 'HRL', 'FANG', 'TT', 'PODD', 'D', 'NFLX', 'WELL', 'CEG', 'EIX', 'AVY', 'GL', 'BRK-B', 'GIS', 'ESS', 'HCA', 'FFIV', 'GLW', 'UPS', 'PG', 'L', 'PEP', 'UNP', 'WRB', 'BSX', 'ZBH', 'MSI', 'CMCSA', 'PANW', 'ISRG', 'NOC', 'CCEP', 'AMP', 'LEN', 'CPT', 'NXPI', 'UDR', 'WMT', 'BMY', 'A', 'LYV', 'ANET', 'CINF', 'STE', 'ACN', 'VTRS', 'WEC', 'DLR', 'FOXA', 'KLAC', 'NCLH', 'NUE', 'ADP', 'UHS', 'KVUE', 'WAT', 'INCY', 'TMUS', 'MRK', 'K', 'FDX', 'DG', 'CCI', 'COIN', 'GWW', 'XOM', 'MCO', 'DDOG', 'LHX', 'ZTS', 'UBER', 'SBAC', 'XYL', 'TRMB', 'MCHP', 'APA', 'HWM', 'GRMN', 'TPR', 'DE', 'ORCL', 'VMC', 'SNA', 'RF', 'WDAY', 'BAX', 'KMI', 'ZS', 'PKG', 'BKR', 'DVN', 'KEYS', 'MLM', 'MTCH', 'COST', 'CLX', 'CTRA', 'LYB', 'PM', 'NWS', 'DOC', 'HAS', 'C', 'WDC', 'ERIE', 'AEP', 'DAY', 'CME', 'APP', 'MU', 'CRWD', 'EFX', 'GILD', 'MSFT', 'RJF', 'PDD', 'HPE', 'TDY', 'ICE', 'BLDR', 'PAYC', 'NKE', 'DXCM', 'IP', 'JNJ', 'FAST', 'CBRE', 'ELV', 'WYNN', 'BA', 'SRE', 'BDX', 'SYY', 'BIIB', 'JBL', 'IBM', 'VRTX', 'DUK', 'CSCO', 'GDDY', 'BEN', 'APTV', 'SBUX', 'ETN', 'GM', 'ROST', 'KDP', 'TSLA', 'FICO', 'MKC', 'MA', 'ASML', 'AXON', 'APO', 'PGR', 'CRL', 'PHM', 'CHTR', 'EPAM', 'GPC', 'CPB', 'CDNS', 'STZ', 'OXY', 'TDG', 'MKTX', 'ALB', 'ODFL', 'ADBE', 'HD', 'WMB', 'RTX', 'BR', 'TYL', 'QCOM', 'GEN', 'AEE', 'KO', 'GOOG', 'KKR', 'IPG', 'TXT', 'AIZ', 'JCI', 'ARM', 'WTW', 'MHK', 'OMC', 'UNH', 'VTR', 'NRG', 'TKO', 'CMS', 'ZBRA', 'PRU', 'LIN', 'FCX', 'DD', 'HAL', 'GE', 'MTD', 'TSN', 'NTAP', 'ATO', 'MMM', 'STX', 'MAA', 'NVR', 'IR', 'NOW', 'WSM', 'CFG', 'ITW', 'MS', 'GNRC', 'EMR', 'SYK', 'AMGN', 'ENPH', 'EMN', 'JNPR', 'RL', 'DHI', 'HSY', 'O', 'VICI', 'AMCR', 'CBOE', 'PH', 'ROL', 'YUM', 'MOS', 'CI', 'WBD', 'MMC', 'ABT', 'PARA', 'AMT', 'FE', 'DASH', 'IT', 'ES', 'EBAY', 'IQV', 'ALL', 'ROK', 'BLK', 'AMAT', 'ADI', 'ABNB', 'AJG', 'OTIS', 'SPGI', 'PPL', 'COP', 'TTD', 'VZ', 'SLB', 'INVH', 'KMX', 'CRM', 'WM', 'LII', 'TPL', 'CNC', 'SNPS', 'GS', 'ECL', 'VST', 'CCL', 'EOG', 'IVZ', 'LRCX', 'EQT', 'SPG', 'BK', 'TRV', 'EW', 'V', 'VLTO', 'OKE', 'MPWR', 'PLTR', 'GEHC', 'NVDA', 'TAP', 'FDS', 'GD', 'INTC', 'SYF', 'SCHW', 'PLD']
tickers = sorted(tickers)

def fetch_and_save_ohlcv(symbol, key, save_path):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={key}&datatype=csv&outputsize=full"
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"[{symbol}] Saved to {save_path}")

load_dotenv(override=True)
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Download stock data for all tickers or a specific symbol.')
    parser.add_argument('--symbol', '-s', type=str, default=None, help='Download only this symbol (e.g. AAPL).')
    args = parser.parse_args()

    if args.symbol:
        ticker_list = [args.symbol.upper()]
    else:
        ticker_list = tickers

    for ticker in ticker_list:
        filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(filepath):  # skip if already downloaded
            fetch_and_save_ohlcv(ticker, API_KEY, filepath)
            time.sleep(0.2)
        else:
            print(f"[{ticker}] Already exists, skipping.")

if __name__ == "__main__":
    main()
