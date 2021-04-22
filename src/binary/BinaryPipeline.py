from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import dataframe

from src.utils.Estimators import get_estimator


def calculate_metrics(trainingSummary):
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
    pass


def create_pipeline(df: dataframe.DataFrame, estimator):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features"
    )
    test, train = df.randomSplit([0.6, 0.4], seed=12345)

    pipeline = Pipeline(stages=[assembler, get_estimator(estimator)])
    # train
    train_model = pipeline.fit(train)

    # training summary
    # calculate_metrics(train_model.summary)

    # make predictions
    predictions = train_model.transform(test)

    # Shows the result.
    return predictions
