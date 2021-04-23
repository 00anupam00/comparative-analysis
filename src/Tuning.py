from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import dataframe

from src.utils.Estimators import get_estimator
from src.multiclass.DataPreProcessor import load_dataset_with_categories


def evaluate_with_train_validation_split(df: dataframe.Dataframe, estimator, evaluator):
    train, test = df.randomSplit([0.9, 0.1], seed=12345)
    algorithm = get_estimator(estimator)
    paramGrid = ParamGridBuilder() \
        .addGrid(algorithm.regParam, [0.1, 0.01]) \
        .addGrid(algorithm.fitIntercept, [False, True]) \
        .addGrid(algorithm.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    tvs = TrainValidationSplit(estimator=algorithm,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    model = tvs.fit(train)
    model.transform(test).select("features", "label", "prediction").show()

    pass

def evaluate_with_cross_validation(df: dataframe.Dataframe, estimator):
    pass


if __name__ == '__main__':
    df = load_dataset_with_categories()
    df.show(10)
    evaluate_with_cross_validation(df, "random_forest", MulticlassClassificationEvaluator())