# Lung Cancer Prediction Data Science Project with MLFlow and Deployment


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/mansipandyaa29/Lung-Cancer-Prediction
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n lungcancer python=3.8 -y
```

```bash
conda activate lungcancer
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

dagshub.init(repo_owner='mansipandyaa29',
             repo_name='Lung-Cancer-Prediction',
             mlflow=True)


