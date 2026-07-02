import pandas as pd
from typing import Tuple

def chronological_split(df: pd.DataFrame, train_frac: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Splits a time-series Dataframe chronologically into train and test sets.
    """

    # Making sure it is sorted 
    df_sorted = df.sort_index()

    # Calculating the row index where the split needs to be made 
    split_idx = int(len(df_sorted) * train_frac)

    # Slicing
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    return train_df, test_df