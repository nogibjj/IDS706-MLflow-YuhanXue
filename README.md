# IDS706-MLflow-YuhanXue

![CICD](https://github.com/nogibjj/DS706-MLflow-YuhanXue/actions/workflows/cicd.yml)

This repo contain project 12. It creates a simple ML model and use MLFlow to log model specs and histories.

## Before runnning
Make sure the following commands are run to install necessary dependenceis and test everything is work fine.
1. `make install`
2. `make test`

Use the following commands to format and lint code: `make format && make lint`.

## How to run
1. In order to start the project, run `python3 main.py`. It first loads the `diabetes.csv` and extract target (i.e. `Outcome` column) and features (i.e. all other columns). It then trains a linear regression model to predict whether a person (data point) has diabetes or not. 
2. After the training is done, run `mlflow ui` to open the user interface and view model histories and test accuracies.
3. View saved models and artifacts in `mlrun` folder.

