import numpy as np
import pandas as pd
import config

class Kalman1D :
    """
        1D Kalman Filter for dynamic hedge ratio estimation
    """

    def __init__(self, beta0: float, P0: float = 1.0, Q: float = config.KALMAN_Q, R: float = config.KALMAN_R):
        self.beta = beta0   # state estimate
        self.P = P0         # Estimate covariance (uncertainty)
        self.Q = Q          
        self.R = R

    def step(self, y_t: float, x_t: float) -> dict:
        """
            Process a single day's and updates beta 
        """

        # Prediction B_t|t-1 = B_{t-1},  P_t|t-1 = P_{t-1} + Q
        beta_pred = self.beta
        P_pred = self.P + self.Q 

        # INNOVATION e_t = y_t - B_t|t-1 · x_t
        expected_y = beta_pred * x_t
        innovation = y_t - expected_y

        # Innovation variance S_t = x_t^2 * P_t|t-1 + R
        innovation_var = (x_t ** 2) * P_pred + self.R

        # Kalman Gain K_t = P_t|t-1 * x_t / S_t
        kalman_gain = (P_pred * x_t) / innovation_var

        # Updating beta and Cov B_t = B_t|t-1 + K_t*e_t, P_t = (1 - K_t*x_t)*P_t|t-1
        self.beta = beta_pred + (kalman_gain * innovation)
        self.P = P_pred * (1 - (kalman_gain * x_t))

        return {
            'beta_t': self.beta,
            'P_t': self.P,
            'innovation': innovation,
            'innovation_var': innovation_var
        } 


    def run(self, y: pd.Series, x: pd.Series, z_window: int = config.ZSCORE_WINDOW) -> pd.DataFrame:
        """
            Run the filter over the entire series
        """

        results = []

        for i in range(len(y)):
            y_t = y.iloc[i]
            x_t = x.iloc[i]

            # Run the Kalman equations for this day
            state = self.step(y_t, x_t)

            # The dynamic spread using today's newly updated beta
            spread_t = y_t - (state['beta_t'] * x_t)

            results.append({
                'Date': y.index[i],
                'beta_t': state['beta_t'],
                'spread': spread_t,
                'innovation': state['innovation'],
                'innovation_var': state['innovation_var']
            })

        df = pd.DataFrame(results).set_index('Date')

        # calculate the rolling Z-score of the dynamic spread
        rolling_mean = df['spread'].rolling(window = z_window).mean()
        rolling_std = df['spread'].rolling(window = z_window).std()
        df['zscore'] = (df['spread'] - rolling_mean) / rolling_std
        
        return df