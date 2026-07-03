import pandas as pd
from config import UNIVERSE, START_DATE, END_DATE, COINT_ALPHA
from data.downloader import get_price_data
from data.cointegration import test_pairs
from models.kalman_filter import Kalman1D
from signals.mean_reversion import MeanReversionSignal
from signals.stat_arb import StatArbSignal
from utils.tca import positions_to_trade, apply_tca
from utils.metrics import compute_returns, walk_forward_report, comparison_report

def main():
    print("--- Project T4: Final Execution ---")

    ## Data and universe selection:
    
    # Flatten the universe to get unique tickers for the downloader
    unique_tickers = list(set([ticker for pair in UNIVERSE for ticker in pair]))
    print(f"Fetching historical data for {len(unique_tickers)} tickers...")
    prices = get_price_data(unique_tickers, START_DATE, END_DATE)

    # Pass the data to the cointegration test
    print("Testing pairs for Cointegration significance...")
    results = test_pairs(UNIVERSE, prices, alpha=COINT_ALPHA)

    # list of actionable pairs
    coint_pairs = results[results['cointegrated'] == True]  #Bool masking
    
    prices_wide = prices.pivot(index='Date', columns='Ticker', values='Close').dropna()

    if coint_pairs.empty:
        print("No cointegration pairs found.")
        return
    
    best_pair = coint_pairs.iloc[0]
    ticker_y, ticker_x = best_pair['ticker1'], best_pair['ticker2']
    print(f"\n Running Strategy on {ticker_y}-{ticker_x} pair")

    y_series = prices_wide[ticker_y]
    x_series = prices_wide[ticker_x]

    ## KALMAN FILTER
    print("Running 1D Kalman Filter")
    kalman = Kalman1D(beta0 = best_pair['beta_ols'])
    kalman_df = kalman.run(y_series, x_series)

    ## SERIES GENERATION
    print("Generating Benchmark (Mean Reversion) Signals")
    bench_signals = MeanReversionSignal()
    bench_positions = bench_signals.generate(kalman_df, prices_wide)

    print("Generating ML StatArb (Walk-Forward) Signals")
    ml_signal = StatArbSignal(ticker_y, ticker_x)
    ml_positions = ml_signal.generate(kalman_df, prices_wide)

    ## TCA & METRICS
    print("Applying Transaction Costs (TCA)")
    
    # Converting posns to trade logs
    bench_trades = positions_to_trade(bench_positions, ticker_y, ticker_x, prices_wide)
    ml_trades = positions_to_trade(ml_positions, ticker_y, ticker_x, prices_wide)

    # Apply Costs
    bench_trades_tca = apply_tca(bench_trades, prices)
    ml_trades_tca = apply_tca(ml_trades, prices)

    # Compute gross returns
    bench_gross = compute_returns(bench_positions, prices_wide, ticker_y, ticker_x)
    bench_net = compute_returns(bench_positions, prices_wide, ticker_y, ticker_x, bench_trades_tca)

    ml_gross = compute_returns(ml_positions, prices_wide, ticker_y, ticker_x)
    ml_net = compute_returns(ml_positions, prices_wide, ticker_y, ticker_x, ml_trades_tca)

    # Generate Reports
    from utils.metrics import sharpe_ratio, sortino_ratio, max_drawdown

    final_stats = {
        'Benchmark': {
            'Sharpe': sharpe_ratio(bench_gross),
            'TCA-Sharpe': sharpe_ratio(bench_net),
            'Sortino': sortino_ratio(bench_net),
            'Max_DD': max_drawdown(bench_net)
        },
        'ML StatArb': {
            'Sharpe': sharpe_ratio(ml_gross),
            'TCA-Sharpe': sharpe_ratio(ml_net),
            'Sortino': sortino_ratio(ml_net),
            'Max_DD': max_drawdown(ml_net)
        }
    }

    print(comparison_report(final_stats))

if __name__ == "__main__":
    main()
    