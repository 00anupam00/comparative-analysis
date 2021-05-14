from pyspark.ml import Estimator, Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.sql import dataframe

from src.utils.Estimators import get_estimator, get_perceptron_estimator, get_estimator_for_multiclass
from src.multiclass.DataPreProcessor import load_dataset_with_categories


def evaluate_with_train_validation_split(df: dataframe.DataFrame, estimator, pipeline, evaluator):
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    algorithm = get_estimator(estimator)
    paramGrid = ParamGridBuilder() \
        .addGrid(algorithm.numTrees, [5, 10, 20]) \
        .addGrid(algorithm.maxDepth, [5, 10]) \
        .addGrid(algorithm.cacheNodeIds, [True, False]) \
        .build()

    tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    model = tvs.fit(train)
    prediction = model.transform(test).select("features", "label", "prediction")
    print("from train validation split hyparam tuner:")
    prediction.show(5)


def evaluate_with_cross_validation(df: dataframe.DataFrame, pipeline, estimator, evaluator):
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    algorithm = get_estimator(estimator)
    paramGrid = ParamGridBuilder() \
        .addGrid(algorithm.numTrees, [5, 10, 20]) \
        .addGrid(algorithm.maxDepth, [5, 10]) \
        .addGrid(algorithm.cacheNodeIds, [True, False]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=2)

    model = crossval.fit(train)
    prediction = model.transform(test)
    print("From cross validation hyperparameters selectors:")
    prediction.show()


def get_pipeline(df, estimator):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features",
        handleInvalid="skip"
    )
    algo = ()
    if estimator.lower() == "perceptron":
        features_col = len(assembler.getInputCols())
        print("Number of features column: ", str(features_col))
        algo = get_perceptron_estimator(features_col)
    else:
        algo = get_estimator_for_multiclass(estimator)
    return Pipeline().setStages([assembler, algo])

#
# if __name__ == '__main__':
#     df = load_dataset_with_categories()
#     estimator = "random_forest"
#     evaluate_with_cross_validation(df, get_pipeline(estimator), estimator, MulticlassClassificationEvaluator())
#     evaluate_with_train_validation_split(df, estimator, get_pipeline(estimator), MulticlassClassificationEvaluator())
