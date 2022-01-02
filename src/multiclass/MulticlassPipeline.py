from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.sql import dataframe

from src.Tuning import evaluate_with_cross_validation, evaluate_with_train_validation_split, get_pipeline
from src.utils.Estimators import get_estimator_for_multiclass, get_perceptron_estimator
from src.utils.PrincipalComponents import get_k


def process_multiclass_pipeline(df: dataframe.DataFrame, estimator):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features_v",
        handleInvalid="skip"
    )

    # PCA
    pca = PCA(k=get_k(), inputCol="features_v", outputCol="features")

    algo = ()
    if estimator.lower() == "perceptron":
        # features_col = len(assembler.getInputCols())
        features_col = pca.getK()
        print("Number of features column: ", str(features_col))
        algo = get_perceptron_estimator(features_col)
    else:
        algo = get_estimator_for_multiclass(estimator)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = df.randomSplit([0.6, 0.4])

    pipeline = Pipeline().setStages([assembler, pca, algo])

    # Tune multiclass
    tdf_cross, tdf_train = tune_multiclass(df, estimator)

    print("Training model ...")
    pipeline_model = pipeline.fit(trainingData)
    pipeline_model.write().overwrite().save("models/multiclass/"+estimator+"/base_model")
    print("Training complete. Base model saved in 'models/multiclass/*'")

    transformed_df = pipeline_model.transform(testData)

    # print("Transformed df:")
    # transformed_df.select("id", "features", "prediction", "rawPrediction", "probability", "label").show(5)

    return transformed_df, tdf_cross, tdf_train


def tune_multiclass(df, estimator):
    print("Tuning model: ", str(estimator))
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    model, tdf_cross = evaluate_with_cross_validation(df, estimator, get_pipeline(df, estimator), evaluator)
    model.write().overwrite().save("models/multiclass/"+estimator+"/cross_validation")
    model, tdf_train = evaluate_with_train_validation_split(df,estimator, get_pipeline(df, estimator), evaluator)
    model.write().overwrite().save("models/multiclass/"+estimator+"/train_validation")
    print("The best models are saved in 'models/multiclass/*'.")
    return tdf_cross, tdf_train
