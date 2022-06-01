from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from taxifare.utils import DistanceTransformer

class TaxiFarePipeline():


    def set_pipeline(self, model):
        pipe_distance = make_pipeline(
            DistanceTransformer(),
            StandardScaler())


        cols = ["pickup_latitude", "pickup_longitude",
                "dropoff_latitude", "dropoff_longitude"]

        feateng_blocks = [
            ('distance', pipe_distance, cols),
        ]

        features_encoder = ColumnTransformer(feateng_blocks, remainder="drop")

        pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('model', model)])


        return pipeline
