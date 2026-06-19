from config import UNIVERSE, START_DATE, END_DATE, COINT_ALPHA
from data.downloader import get_price_data
from data.cointegration import test_pairs

def main():
    print("--- Project T4: Phase 1 ---")

    # Flatten the universe to get unique tickers for the downloader
    unique_tickers = list(set([ticker for pair in UNIVERSE for ticker in pair]))

    print(f"Fetching historical data for {len(unique_tickers)} tickers...")
    prices = get_price_data(unique_tickers, START_DATE, END_DATE)

    # Pass the data to the cointegration test
    print("Testing pairs for Cointegration significance...")
    results = test_pairs(UNIVERSE, prices, alpha=COINT_ALPHA)

    # list of actionable pairs
    coint_pairs = results[results['cointegrated'] == True]  #Bool masking
    print(f"\n[INFO] Found {len(coint_pairs)} cointegrated pairs out of {len(UNIVERSE)}.\n")

    for _, row in coint_pairs.iterrows():
        print(f"✅ {row['pair_id']} (p-value: {row['p_value']:.4f}, initial_beta: {row['beta_ols']:.4f})")

if __name__ == "__main__":
    main()
    