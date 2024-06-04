import pandas as pd
import os
from LungCancerPrediction import logger
import joblib
from sklearn.ensemble import RandomForestClassifier
from LungCancerPrediction.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        y_train = train_y.values.ravel()
        y_test = test_y.values.ravel()

        params = {
        'n_estimators': self.config.n_estimators,
        'max_depth': self.config.max_depth,
        'min_samples_split': self.config.min_samples_split,
        'min_samples_leaf': self.config.min_samples_leaf,
        'max_samples': self.config.max_samples,
        'criterion': self.config.criterion
        }


        model_rf = RandomForestClassifier(**params)
        model_rf.fit(train_x, y_train)

        joblib.dump(model_rf, os.path.join(self.config.root_dir, self.config.model_name))