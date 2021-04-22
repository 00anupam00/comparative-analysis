from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
import neptunecontrib.monitoring.metrics as npt_metrics

def evaluate_binary_classifier(tf_df):
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="prediction")

    # Accuracy
    accuracy = evaluator.evaluate(tf_df)
    areaUnderROC = evaluator.evaluate(tf_df, {evaluator.metricName: "areaUnderROC"})
    print("areaUnderROC: ", str(areaUnderROC))
    print("Accuracy: ", str(accuracy))
    print("Test Error = %g " % (1.0 - accuracy))
    # Confusion Matrix
    tf_df.crosstab("prediction", "label").show(30, False)
    metrics_sklearn(tf_df)
    # todo  precision, fscore


# sklearn Metrics
def metrics_sklearn(tf_df):
    actual = tf_df.select('label').toPandas()
    predicted = tf_df.select('prediction').toPandas()
    acc = round(accuracy_score(actual, predicted), 3) * 100
    print("Accuracy (sklearn): {}%".format(acc))
    print("Confusion Matrix: ")
    cm = confusion_matrix(actual, predicted)
    cm.ravel()
    tn, fp, fn, tp = cm.ravel()
    print("True Negative"
          ": %s \nFalse Positive: %s \nFalse Negative: %s \nTrue Positive: %s" %(tn, fp, fn, tp))
    # plot_precision_recall(actual, predicted, ax=ax)
    # log_loss(actual, predicted)
    # npt_metrics.log_binary_classification_metrics(actual, predicted)
