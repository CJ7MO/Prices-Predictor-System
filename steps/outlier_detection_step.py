import logging

import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step


@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detect and remove outliers using OutlierDetector."""
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}.")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input DataFrame must be a non-null pandas DataFrame.")
    
    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, but got: {type(df)} insted.")
        raise ValueError("Input DataFrame must be a pandas DataFrame.")
    
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' not found in DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=['float', 'int'])

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    print(outliers)
    logging.info("Outlier detection completed.")
    return df_cleaned