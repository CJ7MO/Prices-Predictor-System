from typing import Tuple

import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using a simple train-test split."""
    data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    X_train, X_test, y_train, y_test = data_splitter.split_data(df, target_column)
    return X_train, X_test, y_train, y_test