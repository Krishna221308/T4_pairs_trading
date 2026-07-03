import pandas as pd
from abc import ABC, abstractmethod

class BaseSignal(ABC):
    """
        Abstract base class for all T4 trading signals
    """

    @abstractmethod
    def generate(self, kalman_df: pd.DataFrame, price_df: pd.DataFrame, **kwargs) -> pd.Series:
        """
            Generate trading posns and return a pandas series with posn {+1, 0 -1}
        """
        pass