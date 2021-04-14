from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier

_estimators = {
    'random_forest': RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10),
    'decision_tree': DecisionTreeClassifier(labelCol="label", featuresCol="features"),
    'gbt': GBTClassifier(labelCol="label", featuresCol="features")
}


def get_estimator_keys():
    return _estimators.keys()


def get_estimator(key):
    return _estimators[key]
