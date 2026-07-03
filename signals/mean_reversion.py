import pandas as pd
from signals.base import BaseSignal
import config

class MeanReversionSignal(BaseSignal):
    """
        Pure z-score strategy.
    """

    def generate(self, kalman_df: pd.DataFrame, price_df: pd.DataFrame = None, **kwargs) -> pd.Series:
        current_position = 0
        entry_z = config.ENTRY_Z
        exit_z = config.EXIT_Z

        positions = pd.Series(index=kalman_df.index, data=0)

        for i in range (len(kalman_df)):
            z = kalman_df['zscore'].iloc[i]

            # Entry Logic
            if current_position == 0:
                if z <= -entry_z:
                    current_position = 1        # Buy the spread

                elif z >= entry_z:
                    current_position = -1       # Short the spread

            else:
                if abs(z) <= exit_z:
                    current_position = 0        # Close the trade

            positions.iloc[i] = current_position
        
        return positions
    