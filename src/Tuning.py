from pyspark.ml import Estimator, Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.sql import dataframe

from src.utils.Estimators import get_estimator
from src.multiclass.DataPreProcessor import load_dataset_with_categories


def evaluate_with_train_validation_split(df: dataframe.DataFrame, estimator, evaluator):
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features"
    )

    algorithm = get_estimator(estimator)
    pipeline = Pipeline(stages=[assembler, algorithm])
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

def evaluate_with_cross_validation(df: dataframe.DataFrame, pipeline, estimator: Estimator):
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    algo = get_estimator(estimator)
    paramGrid = ParamGridBuilder() \
        .addGrid(algo.regParam, [0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=2)

    model = crossval.fit(train)
    prediction = model.transform(test)
    print("From cross validation hyperparameters selectors:")
    prediction.show()


if __name__ == '__main__':
    df = load_dataset_with_categories()
    df.show(10)
    evaluate_with_cross_validation(df, "random_forest", MulticlassClassificationEvaluator())