import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Define an abstract method for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self) -> pd.DataFrame:
        """Abstract method for ingesting data from a given file."""
        pass


# Implement a Concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """""Extracts a .zip file and returns the content as a pandas DataFrame."""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("Invalid file type. Please provide a .zip file.")

        # Extract the file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the zip file.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Specify which one to use.")
        
        # Read the CSV file into a DataFrame
        csv_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_path)

        # Return the DataFrame
        return df
    

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """"Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for {file_extension}.")
        


# Example usage:
if __name__ == "__main__":
    # # Specify the file path
    # file_path = r"C:\Users\Administrator\Documents\GitHub\Prices-Predictor-System\data\archive.zip"

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]

    # # Get the appropriate DataIngestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # # Ingest the data and load into a pandas DataFrame
    # df = data_ingestor.ingest(file_path)

    # # Now df contains the DataFrame from the extracted CSV file
    # print(df.head()) # Display the first five rows of the DataFrame
    pass