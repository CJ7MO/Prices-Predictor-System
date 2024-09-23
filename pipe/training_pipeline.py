from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step


from zenml import Model, pipeline


@pipeline(
    model=Model(
        # The name uniquely indentifies this model
        name="prices_predictor"
    ),
)
def ml_pipeline():
    """Define and ent-to-end machine learning pipeline"""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="/Users/Administrator/Documents/GitHub/Prices-Predictor-System/data/archive.zip"
    )

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)


    return filled_data

if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()