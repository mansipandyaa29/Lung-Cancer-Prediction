from LungCancerPrediction.config.configuration import DataIngestionConfig
import shutil


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        source_path = self.config.local_file_path
        destination_path = self.config.root_dir
        shutil.copy(source_path, destination_path)
  