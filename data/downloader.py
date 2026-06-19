import os
import yfinance as yf
import pandas as pd

def get_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads historical daily prices and volumes, saving them locally.
    Returns a long-format DataFrame with columns: [Date, Ticker, Close, Volume]
    """
    cache_dir = "output/cache"
    os.makedirs(cache_dir, exist_ok="True")

    all_data = []

    for ticker in tickers :
        cache_path = os.path.join(cache_dir, f"{ticker}.csv")

        if (os.path.exists(cache_path)) :
            df = pd.read_csv(cache_path, parse_dates = ["Date"])
        else :
            ticker_data = yf.download(ticker, start=start, end=end, progress=False)

            if ticker_data.empty :
                continue
            
            # Flatten multi-index if present
            if isinstance(ticker_data.columns, pd.MultiIndex) :
                ticker_data.columns = ticker_data.columns.droplevel(1)

            # Format to the exact format
            df = ticker_data.reset_index()[['Date', 'Close', 'Volume']].copy()
            df['Ticker'] = ticker
            df = df[['Date', 'Ticker', 'Close', 'Volume']]
            df.to_csv(cache_path, index=False)
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)
