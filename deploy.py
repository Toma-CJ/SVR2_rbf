import numpy as np
from mlflow import MlflowClient
from mlflow.entities import ViewType

metric = "mean_mae"
experiment_id = "98f2615c-0c5e-4ea4-8924-6a742c4fec40"


def find_best_run():
    # find only the runs that are both active and not failed from the experiment
    print("Finding the best run/model...")
    runs = MlflowClient().search_runs(
        experiment_ids=experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    best_rmse = np.inf
    best_run_id = None
    best_params = None
    for r in runs:
        if r.info.status == "FINISHED":
            # ignore runs that didn't finish correctly (e.g. were killed, or setup wrong)
            if metric not in r.data.metrics:
                continue
            cur_id = r.info.run_id
            cur_params = r.data.params
            cur_metric = r.data.metrics[metric]
            if cur_metric < best_rmse:
                best_rmse = cur_metric
                best_run_id = cur_id
                best_params = cur_params
    print("\nFound the best run/model!")
    print(f"run_id: {best_run_id}")
    return best_run_id, best_rmse, best_params


best_id, best_metric, best_params = find_best_run()

print("Found the best model!")
print(f"Model id: {best_id}")
print(f"Metric: {best_metric}")
print("You can now deploy the best model by running the following command:")
print(f"mlflow models serve -m runs:/{best_id}/model --env-manager conda")
