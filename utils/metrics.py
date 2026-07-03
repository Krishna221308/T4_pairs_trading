import pandas as pd
import numpy as np
import config

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, freq: int = config.FREQUENCY) -> float:
    """
        Annualised Sharpe Ratio, Frequency is 252 for daily data
    """

    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free / freq)
    return np.sqrt(freq) * (excess_returns.mean() / excess_returns.std())

def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, freq: int = config.FREQUENCY) -> float:
    """
        Annualised Sortino Ratio (penalizes only downside volatility)
    """
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - (risk_free / freq)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    return np.sqrt(freq) * (excess_returns.mean() / downside_std)

def max_drawdown(returns: pd.Series) -> float:
    """
        Max peak to trough decline as a percentage
    """
    if len(returns) == 0:
        return 0.0
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return drawdowns.min()

def compute_returns(positions: pd.Series, price_df: pd.DataFrame, ticker_y: str, ticker_x: str, trades_tca_df: pd.DataFrame = None) -> pd.Series:
    """
        Computes daily percentage returns of spread startegy. If trades_tcs_df then subtract transaction cost.
    """
    ret_y = price_df[ticker_y].pct_change().fillna(0)
    ret_x = price_df[ticker_x].pct_change().fillna(0)

    shifted_pos = positions.shift(1).fillna(0)

    # Gross daily return of the pair
    # if pos is +1 (Long spread): Long Y, Short X
    # exact opposite for pos -1
    gross_returns = shifted_pos * (ret_y - ret_x)

    # If no TCA return the gross directly
    if trades_tca_df is None or trades_tca_df.empty:
        return gross_returns
    
    # Apply TCA costs
    daily_costs = trades_tca_df.groupby('Date')['Total_Cost'].sum()

    # Convert the absolute values to percentage impacts
    spread_capital = price_df[ticker_y] + price_df[ticker_x]
    percentage_costs = (daily_costs / spread_capital).fillna(0)

    net_returns = gross_returns.sub(percentage_costs, fill_value=0)

    return net_returns

def walk_forward_report(fold_returns: list[pd.Series], aggregate_returns: pd.Series) -> dict:
    """
        Walk-Forward perf report
    """

    # Aggregate stats across the entire out of sample period
    agg_sharpe = sharpe_ratio(aggregate_returns)
    agg_sortino = sortino_ratio(aggregate_returns)
    agg_mdd = max_drawdown(aggregate_returns)

    # Per fold stats to evaluate stability
    fold_sharpes = [sharpe_ratio(fold) for fold in fold_returns]

    # Standard deviation of the per fold sharpes. Lower -> more robust strat
    if len(fold_sharpes) > 1:
        sharpe_stability = np.std(fold_sharpes) 
    else :
        sharpe_stability = 0.0

    return {
        'aggregate_sharpe': agg_sharpe,
        'aggregate_sortino': agg_sortino,
        'aggregate_max_drawdown': agg_mdd,
        'per_fold_shapre': fold_sharpes,
        'sharpe_stability_std': sharpe_stability
    }

def comparison_report(stats_dict: dict) -> str:
    """
        Nicely prints the final comparison table.
    """

    report = "\n" + "="*60 + "\n"
    report += f"{'Strategy Comparison Report':^60}\n"
    report += "="*60 + "\n"
    report += f"{'Metric':<20} | {'Pure Z-Score (Bench)':<20} | {'ML StatArb':<15}\n"
    report += "-"*60 + "\n"

    metrics_to_print = ['Sharpe', 'TCA-Sharpe', 'Sortino', 'Max_DD']

    for metric in metrics_to_print:
        bench_val = stats_dict.get('Benchmark', {}).get(metric, 'N/A')
        ml_val = stats_dict.get('ML StatArb', {}).get(metric, 'N/A')

        if isinstance(bench_val, float): bench_val = f"{bench_val:.3f}"
        if isinstance(ml_val, float): ml_val = f"{ml_val:.3f}"

        report += f"{metric:<20} | {bench_val:<20} | {ml_val:<15}\n"

    report += "="*60 +"\n"
    return report