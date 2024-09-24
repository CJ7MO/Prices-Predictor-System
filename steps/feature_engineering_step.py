import pandas as pd

from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)
from zenml import step


@step
def feature_engineering_step(
    df:pd.DataFrame, 
    strategy:str="log", 
    features: list=None) -> pd.DataFrame:
    """"
    Performs Feature Engineering using FeatureEngineer and selected Stratey
    """
    # Ensure features is a list, even if not provided
    if features is None:
        features = []

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standar_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df