
from influxdb import InfluxDBClient 
import mlflow


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from mlflow.models import infer_signature
from urllib.parse import urlparse


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor


import sys
experiment_name = str(sys.argv[1]) if len(sys.argv) > 1 else sys.exit()

settings = {
    'host': 'influxus.itu.dk',
    'port': 8086,
    'username': 'lsda',
    'password': 'icanonlyread'
    }


client = InfluxDBClient(host=settings['host'], port=settings['port'], username=settings['username'], password=settings['password'])
client.switch_database('orkney')

def set_to_dataframe(resulting_set):
    
    values = resulting_set.raw["series"][0]["values"]
    columns = resulting_set.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) 

    return df

days = 365 

power_set = client.query(
    "SELECT * FROM Generation where time > now()-"+str(days)+"d"
    ) 

wind_set  = client.query(
    "SELECT * FROM MetForecasts where time > now()-"+str(days)+"d and time <= now() and Lead_hours = '1'"
    ) 

power_df = set_to_dataframe(power_set)
wind_df = set_to_dataframe(wind_set)

def clean_data(power_df,wind_df):
    r_power_df = power_df.resample("3h").mean()
    joined_dfs = r_power_df.join(wind_df)
    joined_dfs = joined_dfs.drop(columns=["ANM", "Non-ANM", "Lead_hours", "Source_time", "Direction" ])
    joined_dfs.dropna(inplace=True)
    return joined_dfs

linreg = Pipeline([
    ("Scaler", StandardScaler()),
    ("Linear Regression", LinearRegression())
])

svm = Pipeline([
    ("Scaler", StandardScaler()),
    ("SVR", SVR())
])

sgd = Pipeline([
    ("Scaler", StandardScaler()),
    ("SGD", SGDRegressor())
])

models = [linreg, svm, sgd]

parameters = [
    {
        "Linear Regression__fit_intercept": [True, False]
    },
    {
        "SVR__C": [0.1, 1, 10],
        "SVR__kernel": ["linear", "poly", "rbf", "sigmoid"]
    },
    {
        "SGD__loss": ["squared_error", "huber", "epsilon_insensitive"],
        "SGD__penalty": ["l2","l1"],
        "SGD__fit_intercept": [True, False]
    }
]

mlflow.sklearn.autolog() 

mlflow.set_tracking_uri("http://localhost:5000")



mlflow.set_experiment(experiment_name)
for model, parameter in zip(models, parameters):
    run_name = model.steps[-1][0]
    with mlflow.start_run(run_name=run_name) as run:
    
        joined_dfs = clean_data(power_df,wind_df)

        X = joined_dfs["Speed"].values.reshape(-1,1)
        y = joined_dfs["Total"].values.reshape(-1,1)

        tss = TimeSeriesSplit()
        for train_index, test_index in tss.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        grid_search = GridSearchCV(model, parameter, cv = tss)
        grid_search.fit(X_train, np.ravel(y_train))
        preds = grid_search.predict(X_test)
        
        mae = mean_absolute_error(np.ravel(y_test), preds)
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_param("Best params", grid_search.best_params_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

print("Done and logged to MLflow using latest data") 
    
