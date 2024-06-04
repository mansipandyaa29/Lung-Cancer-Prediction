import os
from LungCancerPrediction import logger
from LungCancerPrediction.entity.config_entity import DataValidationConfig
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True

            # Read the CSV file
            data = pd.read_csv(self.config.data_dir)
            # List all columns
            all_cols = list(data.columns)
        
            # Check if all columns are present in the schema file
            all_schema = self.config.all_schema.keys()

            missing_cols = [col for col in all_cols if col not in all_schema]
            
            with open(self.config.STATUS_FILE, 'w') as f:
                if missing_cols:
                    for col in missing_cols:
                        f.write(f"{col} is not present in schema file\n")
                    validation_status = False
                f.write(f"Validation status: {validation_status}\n")

            return validation_status
        
        except Exception as e:
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation failed due to error: {str(e)}\n")
            raise e
