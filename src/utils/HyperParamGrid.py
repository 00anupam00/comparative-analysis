from pyspark.ml.tuning import ParamGridBuilder

from src.utils.Estimators import get_estimator


def get_param_grid(estimator):
    algorithm = get_estimator(estimator)

    if estimator == "random_forest":
        return ParamGridBuilder() \
            .addGrid(algorithm.numTrees, [5, 10, 20]) \
            .addGrid(algorithm.maxDepth, [5, 10]) \
            .addGrid(algorithm.cacheNodeIds, [True, False]) \
            .build()
    elif estimator == "decision_tree":
        return ParamGridBuilder() \
            .addGrid(algorithm.maxDepth, [3, 6, 9, 12]) \
            .addGrid(algorithm.maxBins, [30, 50]) \
            .addGrid(algorithm.cacheNodeIds, [True, False]) \
            .build()
    elif estimator == "gbt":
        return ParamGridBuilder() \
            .addGrid(algorithm.maxDepth, [3, 6, 9, 12]) \
            .addGrid(algorithm.maxBins, [30, 50]) \
            .addGrid(algorithm.cacheNodeIds, [True, False]) \
            .build()
