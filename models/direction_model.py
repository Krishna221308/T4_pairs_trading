import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import config

def build_features(kalman_df: pd.DataFrame, price_df: pd.DataFrame, ticker_y: str, ticker_x: str) -> pd.DataFrame:
    """
        Returns predictive features from Kalman output and raw prices
    """

    df = kalman_df.copy()

    # Autocorrelation features (is the spread already reverting)
    df['spread lag 1'] = df['spread'].shift(1)
    df['spread lag 2'] = df['spread'].shift(2)
    df['spread lag 3'] = df['spread'].shift(3)
    df['zscore lag 1'] = df['zscore'].shift(1)

    # Volatility Features (Market chaotic ??)
    df['spread_vol_20d'] = df['spread'].rolling(window = 20).std()

    # Kalman Features (Was it an unexpected surprise??)
    df['kalman_surprise'] = df['innovation_var'] / df['innovation_var'].rolling(20).mean()

    # Momentum Features (Trending heavily ??)
    df['mom_y'] = price_df[ticker_y].pct_change(10)
    df['mom_x'] = price_df[ticker_x].pct_change(10)

    # Predict spread tommorow
    df['target_spread_return'] = df['spread'].shift(-1) - df['spread']

    return df.dropna()

def train_ridge_lasso(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
        Training both ridge and lasso models on the training data
    """

    # Initialize models
    ridge = Ridge(alpha = 1.0)
    lasso = Lasso(alpha = 0.01)

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_pred = ridge.predict(X_train)
    lasso_pred = lasso.predict(X_train)

    return {
        'ridge': ridge,
        'lasso': lasso,
        'ridge_r2': r2_score(y_train, ridge_pred),
        'lasso_r2': r2_score(y_train, lasso_pred),
        'ridge_coefs': dict(zip(X_train.columns, ridge.coef_)),         # We keep this to know which coeffs are how important
        'lasso_coefs': dict(zip(X_train.columns, lasso.coef_))          # This to tell if some are even needed or not
    }

def generate_positions(model, features_df: pd.DataFrame, entry_z: float = config.ENTRY_Z, exit_z: float = config.EXIT_Z) -> pd.Series:
    """
        Combing the predictions with Z-Score thresholds to output actual trading positions.
    """

    # Get the models predictions after removing the actual values to avoid lookahead
    X = features_df.drop(columns = ['target_spread_return'])
    predictions = model.predict(X)

    # Create a positions array
    positions = pd.Series(index = features_df.index, data = 0)

    current_position = 0

    for i in range(len(features_df)):
        z = features_df['zscore'].iloc[i]
        pred_spread_change = predictions[i]

        # Entry logic
        if current_position == 0:
            
            # zscore is too high and the model predicts it will fall (SHORT posn)
            if z >= entry_z and pred_spread_change < 0:
                current_position = -1

            # zscore is too low and the model predicts it will rise (LONG posn)
            elif z <= -entry_z and pred_spread_change > 0:
                current_position = 1

        # Exit logic
        else :
            # Exit when zscore reverts back near zero
            if (abs(z) <= exit_z):
                current_position = 0

        positions.iloc[i] = current_position

    return positions
