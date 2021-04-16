from pyspark.ml.evaluation import BinaryClassificationEvaluator


def evaluate_binary_classifier(tf_df):
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="prediction")
    accuracy = evaluator.evaluate(tf_df)
    print("Test Error = %g " % (1.0 - accuracy))
    tf_df.crosstab("prediction", "label").show(30, False)
    # accuracy, precision, fscore
