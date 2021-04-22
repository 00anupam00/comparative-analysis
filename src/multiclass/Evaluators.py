from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import dataframe


def evaluate_multiclass(df: dataframe.DataFrame):
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(df, {evaluator.metricName: "accuracy"})
    weightedPrecision = evaluator.evaluate(df, {evaluator.metricName: "weightedPrecision"})
    weightedFMeasure = evaluator.evaluate(df, {evaluator.metricName: "weightedFMeasure"})
    # fMeasureByLabel = evaluator.evaluate(df, {evaluator.metricName: "fMeasureByLabel"})
    weightedRecall = evaluator.evaluate(df, {evaluator.metricName: "weightedRecall"})
    weightedFalsePositiveRate = evaluator.evaluate(df, {evaluator.metricName: "weightedFalsePositiveRate"})
    weightedTruePositiveRate = evaluator.evaluate(df, {evaluator.metricName: "weightedTruePositiveRate"})
    print("Test Error = %g \nAccuracy: %s \nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % ((1.0 - accuracy), accuracy*100, weightedFalsePositiveRate, weightedTruePositiveRate, weightedFMeasure, weightedPrecision, weightedRecall))
