from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import dataframe

from src.utils.Estimators import get_estimator, get_estimator_for_multiclass


def process_multiclass_pipeline(df: dataframe.DataFrame, estimator):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features",
        handleInvalid="skip"
    )

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    estimator = get_estimator_for_multiclass(estimator)

    pipeline = Pipeline().setStages([assembler, estimator])
    pipeline_model = pipeline.fit(trainingData)

    transformed_df = pipeline_model.transform(testData)

    # print("Schema after transformation:")
    print("Transformed df:")
    transformed_df.printSchema()
    transformed_df.select("id", "prediction", "rawPrediction", "probability", "label").show(5)

    return transformed_df
