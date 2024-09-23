import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropiate DataIngestor."""
    # Determine the file extension
    file_extension = ".zip"

    # Get the appropiate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a pandas DataFrame
    df = data_ingestor.ingest(file_path)
    return df