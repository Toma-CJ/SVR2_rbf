import mlflow
#from mlflow.models import infer_signature
import pickle

#512b3ceadaea4908bd08331ee55a7d93
#runs:/<run_id>/<artifact_path>

#loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
#load model
#model_uri = f"models:/{model_name}/{model_version}"

#model.predict()

svm = Pipeline([
    ("Scaler", StandardScaler()),
    ("SVR", SVR(C= 1,kernel="rbf"))
])

joined_dfs = clean_data(power_df,wind_df)

# X = joined_dfs["Speed"].values.reshape(-1,1)
# y = joined_dfs["Total"].values.reshape(-1,1)

X = joined_dfs["Speed"]
y = joined_dfs["Total"]
tss = TimeSeriesSplit()
for train_index, test_index in tss.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

svm.fit((X_train, np.ravel(y_train)))
preds = svm.predict(X_test)
mae = mean_absolute_error(np.ravel(y_test), preds)

print(f"MAE: {mae}")