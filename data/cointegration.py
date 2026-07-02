import numpy as np
import config
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import os

def half_life(spread: pd.Series) -> float:
    """
        Calculate the half life of a mean reversion for a spread
        Fits an AR(1) model: change in spread = a*spread_{t-1} + ephsilon
        Returns the half life in days. 
        Returns NaN if it doesn't exist.
    """
    
    # Calculate day to day changes in the spread.
    spread_diff = spread.diff().dropna()

    # We need the previous day's spread aligned with today's change
    spread_lagged = spread.shift(1).dropna()

    # Align the two series
    spread_diff = spread_diff.loc[spread_lagged.index]

    if len(spread_diff) == 0:
        return np.nan
    
    # Run OLS Regression without an intercept using statsmodel OLS
    model = sm.OLS(spread_diff, spread_lagged).fit()
    lambda_param = model.params.iloc[0]

    if lambda_param >= 0:
        return np.nan
    
    # Half-life formula
    return -np.ln(2)/lambda_param


def test_pairs(pairs: list[tuple], price_df: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    """
    Tests asset pairs for cointegration using the Engle-Granger two-step method.
    Returns a DataFrame of pairs and their structural parameters.
    """
    results = []
    wide_prices = price_df.pivot(index='Date', columns='Ticker', values='Close').dropna()

    for pair in pairs :
        ticker1, ticker2 = pair

        if ticker1 not in wide_prices.columns or ticker2 not in wide_prices :
            continue

        y = wide_prices[ticker1]
        x = wide_prices[ticker2]

        # Engle-Granger Cointegration Test (Augmented Dickey-Fuller on residuals) 
        score, p_value, _ = coint(y,x)

        # OLS Regression to capture initial hedge ratio parameters
        x_with_const = sm.add_constant(x)
        ols_model = sm.OLS(y, x_with_const).fit()
        alpha_ols = ols_model.params['const']
        beta_ols = ols_model.params[ticker2]

        # Calculate the historical spread
        spread = y - (alpha_ols + beta_ols * x)

        # Using half-life 
        hl = half_life(spread)

        is_cointegrated = ((p_value < alpha) and (pd.notna(hl)) and (config.HALF_LIFE_MIN <= hl) and (config.HALF_LIFE_MAX >= hl))

        pair_id = f"{ticker1}_{ticker2}"

        results.append({
            'pair_id': pair_id,
            'ticker1': ticker1,
            'ticker2': ticker2,
            'p_value': p_value,
            'beta_ols': beta_ols,
            'alpha_ols': alpha_ols,
            'half-life': hl,
            'cointegrated': is_cointegrated
        })

    results_df = pd.DataFrame(results)

    # Saving a master record of the relationships
    os.makedirs("output", exist_ok=True)
    results_df.to_csv("output/cointegration_results.csv", index=False)

    return results_df
