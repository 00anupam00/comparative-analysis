from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import dataframe

from src import Estimators


def create_pipeline(df: dataframe.DataFrame, estimator):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features"
    )

    pipeline = Pipeline(stages=[assembler, Estimators.get_estimator(estimator)])
    pipeline_model = pipeline.fit(df)
    predictions = pipeline_model.transform(df)

    # todo Evaluators and hyper tuning is the next step

    print("Schema after transformation:")
    predictions.printSchema()

    # Shows the result.

    return predictions
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)
