from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step


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
    print(raw_data)   

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    print(engineered_data)

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")
    print(clean_data)

    return clean_data

if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()