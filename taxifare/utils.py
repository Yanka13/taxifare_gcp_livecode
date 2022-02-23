from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from google.cloud import storage

def minkowski_distance(df, p,
                       start_lat="pickup_latitude",
                       start_lon="pickup_longitude",
                       end_lat="dropoff_latitude",
                       end_lon="dropoff_longitude"):
    x1 = df[start_lon]
    x2 = df[end_lon]
    y1 = df[start_lat]
    y2 = df[end_lat]
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)

class DistanceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, distance_type="euclidian", **kwargs):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.distance_type == "euclidian":
            X["distance"] = minkowski_distance(X, p=2)
        if self.distance_type == "manhattan":
            X["distance"] = minkowski_distance(X, p=1)
        return X[["distance"]]

    def fit(self, X, y=None):
        return self


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def upload_model_to_gcp():

    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    bucket_name = "taxifare-805-yannis"
    # The path to your file to upload
    source_file_name = "model.joblib"
    # The ID of your GCS object
    destination_blob_name = "models/best_model.joblib"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


