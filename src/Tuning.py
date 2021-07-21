from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.tuning import TrainValidationSplit, CrossValidator
from pyspark.sql import dataframe

from src.utils.Estimators import get_perceptron_estimator, get_estimator
from src.utils.HyperParamGrid import get_param_grid
from src.utils.PrincipalComponents import get_k


def evaluate_with_train_validation_split(df: dataframe.DataFrame, estimator, pipeline, evaluator):
    print("Tuning with train validation split...")
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    paramGrid = get_param_grid(estimator)

    tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    model = tvs.fit(train)
    tf_df = model.transform(test)
    # tf_df.select("features", "label", "prediction").show(5)

    return model.bestModel, tf_df


def evaluate_with_cross_validation(df: dataframe.DataFrame, estimator, pipeline, evaluator):
    print("Tuning with cross validation...")
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    paramGrid = get_param_grid(estimator)

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=2)

    model = crossval.fit(train)
    tf_df = model.transform(test)
    # tf_df.select("features", "label", "prediction").show(5).show()

    return model.bestModel, tf_df


def get_pipeline(df, estimator):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features_v",
        handleInvalid="skip"
    )
    
    pca = PCA(k=get_k(), inputCol="features_v", outputCol="features")
    
    algo = ()
    if estimator.lower() == "perceptron":
        features_col = pca.getK()
        print("Number of features column: ", str(features_col))
        algo = get_perceptron_estimator(features_col)
    else:
        algo = get_estimator(estimator)
    return Pipeline().setStages([assembler, pca, algo])
