from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
import neptunecontrib.monitoring.metrics as npt_metrics
from scikitplot.metrics import plot_roc
from matplotlib import pyplot as plt

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
    recall = calculate_recall(tp, fn)
    precision = calculate_precision(tp, fp)
    f_score = calculate_f_score(recall, precision)
    print("Recall : %s \nPrecision: %s \n F_score: %s " %(recall, precision, f_score))
    print("Plot of Area under ROC:")
    # plot_area_under_roc(actual, predicted)


def calculate_recall(tp, fn):
    return tp / (tp + fn)


def calculate_precision(tp, fp):
    return tp/(fp + tp)

def calculate_f_score(recall, precision):
    return (2 * recall * precision)/(recall + precision)

def plot_area_under_roc(actual, predicted):
    fig, ax = plt.subplots()
    plot_roc(actual, predicted, ax=ax)
