import pandas as pd
from statsmodels.tsa.stattools import adfuller
from signals.base import BaseSignal
from models.direction_model import build_features, train_ridge_lasso, generate_positions
import config

class StatArbSignal(BaseSignal):
    """
        ML-gated z-score strategy. Using a pre-trained ridge-lasso model to filter trades.
    """

    def __init__(self, ticker_y: str, ticker_x: str):
        """
            Initialising trained model and the ticker names to build the features
        """

        self.ticker_y = ticker_y
        self.ticker_x = ticker_x

    def generate(self, kalman_df: pd.DataFrame, price_df: pd.DataFrame, **kwargs) -> pd.Series:
        # Build the features using the build_features function from direction_model.py
        features_df = build_features(kalman_df, price_df, self.ticker_y, self.ticker_x)

        # Since the build_features might return NaN values to get rid of those during the backtesting we do 
        final_positions = pd.Series(index=kalman_df.index, data=0)
        
        refit_interval = config.REFIT_INTERVAL
        lookback_coint = config.LOOKBACK_COINT

        start_idx = lookback_coint

        for t in range(start_idx, len(kalman_df), refit_interval):
            # defining the current train window
            train_features = features_df.iloc[:t]

            # To predict
            test_end = min (t + refit_interval, len(features_df))
            test_features = features_df.iloc[t:test_end]

            if len(test_features) == 0:
                break

            # Mid walk coint
            recent_spread = kalman_df['spread'].iloc[t - lookback_coint:t]

            # Run ADF
            adf_stat, p_value, _, _, _, _ =adfuller(recent_spread.dropna())

            if p_value >= 0.10:
                #Cointegration broken halt for this period
                continue

            # Model still works (retraining)
            X_train = train_features.drop(columns=['target_spread_returns'])
            y_train = train_features['target_spread_returns']

            models = train_ridge_lasso(X_train ,y_train)
            current_model = models['ridge']

            window_positions = generate_positions(current_model, test_features)

            final_positions.update(window_positions)
        
        return final_positions