import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from LungCancerPrediction.utils.common import save_json
from LungCancerPrediction.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import dagshub


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initialize_dagshub(self):
        dagshub.init(repo_owner='mansipandyaa29',repo_name='Lung-Cancer-Prediction',mlflow=True)
    
    def eval_metrics(self,actual, pred):
        precision = precision_score(actual, pred, average = 'micro')
        recall = recall_score(actual, pred, average = 'micro')
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred, average = 'micro')
        r2 = r2_score(actual, pred)
        return precision, recall, accuracy, f1, r2

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        self.initialize_dagshub()

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (precision, recall, accuracy, f1, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("r2", r2)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")