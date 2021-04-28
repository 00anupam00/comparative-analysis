from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier, NaiveBayes, LinearSVC

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
    # input layer of size 115 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = [115, 5, 4, 3]

    _estimators = {
        'random_forest': RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10),
        'decision_tree': DecisionTreeClassifier(labelCol="label", featuresCol="features"),
        'gbt': GBTClassifier(labelCol="label", featuresCol="features"),  # only supports binary classification

        ## extras
        'nb': NaiveBayes(smoothing=1.0, modelType="multinomial"), # requires non-negative features
        'perceptron' : MultilayerPerceptronClassifier(labelCol='label', featuresCol='features', maxIter=100, layers=layers, blockSize=128, seed=1234),
        "lsvc" : LinearSVC(maxIter=10, regParam=0.1) # only supports binary classification
    }
    return _estimators


def get_estimator_keys():
    return estimators_for_classifiers().keys()


def get_estimator(key):
    return estimators_for_classifiers()[key]

def get_estimator_for_multiclass(key):
    return estimators_for_multiclass()[key]
