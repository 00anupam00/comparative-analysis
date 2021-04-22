from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier
from pyspark.mllib.classification import NaiveBayes

_estimators = dict()

def estimators_for_classifiers():
    global _estimators
    _estimators = {
        'random_forest': RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10),
        'decision_tree': DecisionTreeClassifier(labelCol="label", featuresCol="features"),
        'gbt': GBTClassifier(labelCol="label", featuresCol="features")
    }
    return _estimators

def estimators_for_multiclass():
    global _estimators
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = [4, 5, 4, 3]

    _estimators = {
        'random_forest': RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10),
        'decision_tree': DecisionTreeClassifier(labelCol="label", featuresCol="features"),
        'gbt': GBTClassifier(labelCol="label", featuresCol="features"),
        'nb': NaiveBayes(smoothing=1.0, modelType="multinomial"),
        'perceptron' : MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    }
    return _estimators


def get_estimator_keys():
    return estimators_for_classifiers().keys()


def get_estimator(key):
    return estimators_for_classifiers()[key]
