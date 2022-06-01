from sklearn.ensemble import RandomForestRegressor
import joblib
from taxifare.data import clean_df, holdout, get_data_using_pandas
from taxifare.mlflow_tracker import MLFlowBase
from taxifare.pipeline import TaxiFarePipeline
from taxifare.utils import compute_rmse, upload_model_to_gcp
from taxifare.mlflow_tracker import MLFlowBase

class Trainer(MLFlowBase):


    def __init__(self):
        super().__init__(
            "[UK] [London] [Yanka13] TaxiFare Livecode #900 v1", "https://mlflow.lewagon.ai/")

    def save_model(self):
        #save the model
        joblib.dump(self.pipe, "model.joblib")

    def score_rmse(self):

        y_pred = self.pipe.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        self.rmse = rmse
        return rmse


    def train(self, line_count):


        #launch a run
        print('Training start.... ')
        self.mlflow_create_run()

        #get the data
        df = get_data_using_pandas(line_count)
        df = clean_df(df)
        X_train, X_test, y_train, y_test = holdout(df)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        #choose a model
        model_params = dict(
            n_estimators=100,
            max_depth=1)


        model = RandomForestRegressor()
        model.set_params(**model_params)

        # log model and hyperparam
        self.mlflow_log_param("model_name", "RandomForestRegressor")
        self.mlflow_log_param("n_estimators", 100)
        self.mlflow_log_param("line_count", line_count)
        print(f"Training with {line_count} rows in our dataset")


        # create the pipeline
        taxi_pipeline = TaxiFarePipeline()
        pipeline = taxi_pipeline.set_pipeline(model)


        #fit the pipeline
        pipeline.fit(X_train, y_train)
        self.pipe = pipeline

        #output the score on the test set
        self.score_rmse()

        # log rmse on mlflow
        self.mlflow_log_metric("rmse", self.rmse)
        print(f"rmse on test set is {self.rmse}")

        #save the model
        self.save_model()

        #save the model on our bucket
        upload_model_to_gcp()


        return self.pipe


# We call here everything we need to do a training
if __name__ == "__main__":

    trainer = Trainer()
    trainer.train(100)
    print("Good job google!")


