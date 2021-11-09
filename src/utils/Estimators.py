from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier, LogisticRegression, LinearSVC, NaiveBayes, FMClassifier

_estimators = dict()


def estimators_for_classifiers():
    global _estimators
    # specify layers for the neural network:
    # input layer of size 115 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)

    _estimators = {
        'random_forest': RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10),
        'decision_tree': DecisionTreeClassifier(labelCol="label", featuresCol="features"),
        'gbt': GBTClassifier(labelCol="label", featuresCol="features"),  # only supports binary classification
        # 'lr' : LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8), # doesn't work
        # 'lsvc' : LinearSVC(maxIter=10, regParam=0.1), # doesn't work
        # 'nb' : NaiveBayes(smoothing=1.0, modelType="multinomial"), # doesn't work
        # 'fm' : FMClassifier(labelCol="indexedLabel", featuresCol="scaledFeatures", stepSize=0.001) # doesn't work
    }
    return _estimators


def get_estimator_keys():
    return estimators_for_classifiers().keys()


def get_estimator(key):
    return estimators_for_classifiers()[key]


def get_estimator_for_multiclass(key):
    return estimators_for_classifiers()[key]


def get_perceptron_estimator(length_of_features):
    layers = [length_of_features, 5, 4, 4]
    return MultilayerPerceptronClassifier(labelCol='label', featuresCol='features', maxIter=100,
                                          layers=layers, blockSize=128, seed=1234)
