import os
from LungCancerPrediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from LungCancerPrediction.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self,data):
        data = data.drop(columns=['index', 'Patient Id'], axis=1)
        data["Level"] = data["Level"].replace({'High': 2, 'Medium': 1, 'Low': 0})
        data = data.infer_objects(copy=False)
        return data

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        data = self.transform(data)
        
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        