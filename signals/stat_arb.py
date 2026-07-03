import pandas as pd
from signals.base import BaseSignal
from models.direction_model import build_features, generate_positions
import config

class StatArbSignal(BaseSignal):
    """
        ML-gated z-score strategy. Using a pre-trained ridge-lasso model to filter trades.
    """

    def __init__(self, trained_model, ticker_y: str, ticker_x: str):
        """
            Initialising trained model and the ticker names to build the features
        """

        self.model = trained_model
        self.ticker_y = ticker_y
        self.ticker_x = ticker_x

    def generate(self, kalman_df: pd.DataFrame, price_df: pd.DataFrame, **kwargs) -> pd.Series:
        # Build the features using the build_features function from direction_model.py
        features_df = build_features(kalman_df, price_df, self.ticker_y, self.ticker_x)

        # Now filter the signals using the ML model
        positions = generate_positions(self.model, features_df)

        # Since the build_features might return NaN values to get rid of those during the backtesting we do 
        final_positions = pd.Series(index=kalman_df.index, data=0)
        final_positions.update(positions)

        return final_positions