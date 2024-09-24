import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract base class for Outlier Detection Strategy
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers in the specified DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to detect outliers in.

        Returns:
        pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        pass


# Concrete Strategy for Z-score based Outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using Z-score method.")
        z_scores= np.abs((df-df.mean())/df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        return outliers
    

# Concrete Strategy for IQR based outlier detection
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method.")
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        outliers = (df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))
        logging.info("Outliers detected using IQR method.")
        return outliers
    

# Context Class for Outlier Detection and Handling
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Switching outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method: '{method}'. Skipping outlier handling.")
            return df
        
        logging.info("Outlier handling completed.")
        return df_cleaned
    
    def visualize_outliers(self, df: pd.DataFrame, features:str):
        logging.info(f"Visualizing outliers for features: {features}.")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of: {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # data_path = r'C:\Users\Administrator\Documents\GitHub\Prices-Predictor-System\extracted_data\AmesHousing.csv'
    # df = pd.read_csv(data_path)
    # df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    # outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # # Detect and handle outliers
    # outliers = outlier_detector.detect_outliers(df_numeric)
    # df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")

    # print(df_cleaned.shape)
    # # Visualize outliers in specific features
    # outlier_detector.visualize_outliers(df_cleaned, features=["SalePrice", "Gr Liv Area"])
    pass

#    SalePrice  Gr Liv Area
# 0     200000        1500
# 1     300000        2000
# 2     400000        2500
# 3     500000        3000
# 4    1000000        4000
# 5    2000000       10000


#    SalePrice  Gr Liv Area
# 0     200000        1500
# 1     300000        2000
# 2     400000        2500
# 3     500000        3000
# 4    1000000        4000
# 5    1000000        4000