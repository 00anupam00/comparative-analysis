from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.tuning import ParamGridBuilder

from src.utils.Estimators import get_estimator


def get_param_grid(estimator):

    if estimator == "random_forest":
        algorithm = get_estimator(estimator)
        return ParamGridBuilder() \
            .addGrid(algorithm.numTrees, [5, 10, 20]) \
            .addGrid(algorithm.maxDepth, [5, 10]) \
            .addGrid(algorithm.cacheNodeIds, [True, False]) \
            .build()
    elif estimator == "decision_tree":
        algorithm = get_estimator(estimator)
        return ParamGridBuilder() \
            .addGrid(algorithm.maxDepth, [3, 6, 9, 12]) \
            .addGrid(algorithm.maxBins, [30, 50]) \
            .addGrid(algorithm.cacheNodeIds, [True, False]) \
            .build()
    elif estimator == "gbt":
        algorithm = get_estimator(estimator)
        return ParamGridBuilder() \
            .addGrid(algorithm.maxDepth, [3, 6, 9, 12]) \
            .addGrid(algorithm.maxBins, [30, 50]) \
            .addGrid(algorithm.cacheNodeIds, [True, False]) \
            .build()
    elif estimator == "perceptron":
        layers = [23, 10, 6, 4]
        algorithm = MultilayerPerceptronClassifier(labelCol='label', featuresCol='features', maxIter=100,
                                          layers=layers, blockSize=128, seed=1234)
        return ParamGridBuilder() \
            .addGrid(algorithm.maxIter, [75]) \
            .addGrid(algorithm.blockSize, [150, 200]) \
            .addGrid(algorithm.layers, [[23, 15, 10, 4]]) \
            .build()
    elif estimator == "lr":
        algorithm = get_estimator(estimator)
        return ParamGridBuilder() \
            .addGrid(algorithm.regParam, [0.1, 0.01]) \
            .addGrid(algorithm.fitIntercept, [False, True]) \
            .addGrid(algorithm.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
    elif estimator == "nb":
        algorithm = get_estimator(estimator)
        return ParamGridBuilder() \
            .addGrid(algorithm.smoothing, [1.0, 2.0]) \
            .build()
    elif estimator == "fm":
        algorithm = get_estimator(estimator)
        return ParamGridBuilder() \
            .addGrid(algorithm.maxIter, [75, 100, 150]) \
            .addGrid(algorithm.fitIntercept, [False, True]) \
            .addGrid(algorithm.fitLinear, [False, True]) \
            .build()