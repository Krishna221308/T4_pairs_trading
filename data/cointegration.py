import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import os

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
        is_cointegrated = p_value < alpha

        # OLS Regression to capture initial hedge ratio parameters
        x_with_const = sm.add_constant(x)
        ols_model = sm.OLS(y, x_with_const).fit()
        alpha_ols = ols_model.params['const']
        beta_ols = ols.model.params[ticker2]

        pair_id = f"{ticker1}_{ticker2}"

        results.append({
            'pair_id': pair_id,
            'ticker1': ticker1,
            'ticker2': ticker2,
            'p_value': p_value,
            'beta_ols': beta_ols,
            'alpha_ols': alpha_ols,
            'cointegrated': is_cointegrated
        })

    results_df = pd.DataFrame(results)

    # Saving a master record of the relationships
    os.makedirs("output", exist_ok=True)
    results_df.to_csv("output/cointegration_results.csv", index=False)

    return results_df
