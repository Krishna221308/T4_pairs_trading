import pandas as pd
import numpy as np
import config

def slippage_cost(trade_price: float, bps: float = config.SLIPPAGE_BPS) -> float:
    """
        Calculate half-spread cost per unit traded
    """
    return trade_price *(bps/10000.0)

def market_impact(trade_size: float, adv: float, daily_vol: float, gamma: float = config.IMPACT_GAMMA) -> float:
    """
        Square root market impact model.
    """

    if adv == 0 or pd.isna(adv) :
        return 0.0
    return daily_vol * gamma * np.sqrt(abs(trade_size) / adv)

def positions_to_trade(positions: pd.Series, ticker_y: str, ticker_x: str, price_df: pd.DataFrame) -> pd.DataFrame:
    """
        Converts position series into trade logs
    """

    # Calculate day to day changes
    pos_diff = positions.diff().fillna(0)

    # Only keeping the days traded
    trade_days = pos_diff[pos_diff != 0].index
    trades = []
    for date in trade_days:
        change = pos_diff.loc[date]
        price_y = price_df.loc[date, ticker_y]
        price_x = price_df.loc[date, ticker_x]

        # Log y
        trades.append({
            'Date': date,
            'Ticker': ticker_y,
            'Side': np.sign(change),
            'Size': abs(change),
            'Price': price_y
        })

        # Log x
        trades.append({
            'Date': date,
            'Ticker': ticker_x,
            'Side': -np.sign(change),
            'Size': abs(change),
            'Price': price_x
        })

    return pd.DataFrame(trades)

def apply_tca(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
        Apply transaction costs to trade logs
    """

    if trades_df.empty:
        return trades_df
    
    # Calculate rolling ADV and volatility on the raw price data
    price_df = price_df.sort_values(by=['Ticker', 'Date'])
    price_df['ADV'] = price_df.groupby('Ticker')['Volume'].transform(lambda x:x.rolling(20).mean())
    price_df['Daily_Vol'] = price_df.groupby('Ticker')['Close'].transform(lambda x:x.pct_change().rolling(20).std())

    trades_df = trades_df.merge(
        price_df[['Date', 'Ticker', 'ADV', 'Daily_Vol']],
        on=['Date', 'Ticker']
        how='left'
    )

    # Calculate slippage
    trades_df['Slippage'] = trades_df.apply(
        lambda row: slippage_cost(row['Price']*row['Size'], axis=1)
    )

    # Calculate Market Impact
    trades_df['Impact'] = trades_df.apply(
        lambda row: market_impact(row['Size'], row['ADV'], row['Daily_Vol']) * row['Price'] * row['Size'], axis=1
    )

    trades_df['Total_Cost'] = trades_df['Slippage'] + trades_df['Impact']

    return trades_df
