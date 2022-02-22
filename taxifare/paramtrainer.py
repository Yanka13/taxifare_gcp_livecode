from taxifare.trainer import Trainer
from taxifare.data import get_data, clean_df, holdout
from taxifare.pipeline import TaxiFarePipeline
from taxifare.utils import compute_rmse


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import joblib

class ParamTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.i = 0


    def train(self, params):

        grid_search_models = dict()
        for model_name, param_dict in params.items():
            print(model_name, param_dict)

            if model_name == "random_forest":
                model = RandomForestRegressor()
            else:
                model = LinearRegression()

            #launch a run
            self.mlflow_create_run()

            #log model and hyper params for gridsearch
            self.mlflow_log_param("model_name", model_name)
            self.mlflow_log_param("line_count", param_dict["line_count"])
            for hyper_param_name, hyper_param_value in param_dict["hyper_params"].items():
                self.mlflow_log_param(hyper_param_name, str(hyper_param_value))

            #get the data
            df = get_data(line_count=param_dict["line_count"])
            df = clean_df(df)
            X_train, X_test, y_train, y_test = holdout(df)

            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

            # create the pipeline
            taxi_pipeline = TaxiFarePipeline()
            pipeline = taxi_pipeline.set_pipeline(model)

            #do a grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_dict["hyper_params"],
                cv=5
            )

            grid_search.fit(X_train, y_train)
            self.grid = grid_search
            #log score on best grid search model
            y_pred = self.grid.best_estimator_.predict(X_test)
            rmse = compute_rmse(y_pred, self.y_test)
            self.rmse = rmse
            self.mlflow_log_metric("rmse", rmse)

            grid_search_models[model_name] = grid_search


            joblib.dump(self.grid.best_estimator_, f"../saved_models/{model_name}_model_{self.i}.joblib")
            self.i+=1
        return grid_search_models




