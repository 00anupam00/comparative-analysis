from pyspark.ml.classification import RandomForestClassifier

_estimators = {
    'random_forest': RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
}


def get_estimator_keys():
    return _estimators.keys()


def get_estimator(key):
    return _estimators[key]
