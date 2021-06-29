from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import dataframe

from src.Tuning import evaluate_with_cross_validation, evaluate_with_train_validation_split, get_pipeline
from src.utils.Estimators import get_estimator_for_multiclass, get_perceptron_estimator


def process_multiclass_pipeline(df: dataframe.DataFrame, estimator):
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

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # feature_selector =  get_selector("features", "label")

    pipeline = Pipeline().setStages([assembler, algo])

    # Tune multiclass
    tdf_cross, tdf_train = tune_multiclass(df, estimator)

    pipeline_model = pipeline.fit(trainingData)
    transformed_df = pipeline_model.transform(testData)

    # print("Schema after transformation:")
    print("Transformed df:")
    transformed_df.printSchema()
    transformed_df.select("id", "features", "prediction", "rawPrediction", "probability", "label").show(5)

    return transformed_df, tdf_cross, tdf_train


def tune_multiclass(df, estimator):
    print("Tuning model: ", str(estimator))
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    tdf_cross = evaluate_with_cross_validation(df, estimator, get_pipeline(df, estimator), evaluator)
    tdf_train = evaluate_with_train_validation_split(df,estimator, get_pipeline(df, estimator), evaluator)
    return tdf_cross, tdf_train
