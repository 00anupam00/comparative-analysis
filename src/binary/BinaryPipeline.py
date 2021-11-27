from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.feature import PCA
from pyspark.sql import dataframe

from src.Tuning import evaluate_with_train_validation_split, get_pipeline, evaluate_with_cross_validation
from src.utils.Estimators import get_estimator
from src.utils.PrincipalComponents import get_k
from datetime import datetime


def calculate_metrics(trainingSummary):
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
    pass


def process_binary_pipeline(df: dataframe.DataFrame, estimator):
    pipeline = Pipeline()
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features_v"
    )

    test, train = df.randomSplit([0.6, 0.4], seed=12345)

    if estimator == "nb":
        scaler = MinMaxScaler(inputCol="features_v", outputCol="scaledFeatures")
        pca = PCA(k=get_k(), inputCol="scaledFeatures", outputCol="features")
        pipeline = Pipeline(stages=[assembler, scaler, pca, get_estimator(estimator)])
    else:
        pca = PCA(k=get_k(), inputCol="features_v", outputCol="features")
        pipeline = Pipeline(stages=[assembler, pca, get_estimator(estimator)])

    print("Training model ...")
    trainStartTime = datetime.now()
    train_model = pipeline.fit(train)
    train_model.write().overwrite().save("models/binary/"+estimator+"/base_model")
    print("Training complete. The base model is saved in 'models/binary/*'.")
    trainEndTime = datetime.now()
    trainTime = str(trainEndTime - trainStartTime)
    print("Total time required for training: ",trainTime)
    # make predictions
    tf_df = train_model.transform(test)

    # training summary
    # calculate_metrics(train_model.summary)

    # tune pipeline before fit
    print("Tuning binary pipeline...")
    tdf_cross, tdf_train = tune_binary(df, estimator)
    return tf_df, tdf_cross, tdf_train


def tune_binary(df, estimator):
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")

    # validation split
    model, validationSplit_tdf = evaluate_with_train_validation_split(df, estimator, get_pipeline(df, estimator), evaluator)
    model.write().overwrite().save("models/binary/"+estimator+"/train_validation")

    # cross validation fit
    model, cross_valid_tdf = evaluate_with_cross_validation(df, estimator, get_pipeline(df, estimator), evaluator)
    model.write().overwrite().save("models/binary/"+estimator+"/cross_validation")

    print("The best models are saved in 'models/binary/*'.")

    return cross_valid_tdf, validationSplit_tdf
