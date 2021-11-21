from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, HashingTF, StringIndexer, OneHotEncoder
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


def process_binary_pipeline(df: dataframe.DataFrame, estimator, raw_data):
    pipeline = Pipeline()
    if raw_data:
        stages = prepare_stages_for_raw_data()
        pipeline.setStages(stages)
    else:
        assembler = VectorAssembler(
            inputCols=[x for x in df.columns if x != "label"],
            outputCol="features_v"
        )
        # PCA
        pca = PCA(k=get_k(), inputCol="features_v", outputCol="features")  # todo can this be omitted?
        pipeline.setStages([assembler, pca, get_estimator(estimator)])

    test, train = df.randomSplit([0.7, 0.3], seed=12345)

    print("Training model ...")
    trainStartTime = datetime.now()
    train_model = pipeline.fit(train)
    train_model.write().overwrite().save("models/binary/" + estimator + "/base_model")
    print("Training complete. The base model is saved in 'models/binary/*'.")
    trainEndTime = datetime.now()
    trainTime = str(trainEndTime - trainStartTime)
    print("Total time required for training: ", trainTime)
    # make predictions
    tf_df = train_model.transform(test)

    # training summary
    # calculate_metrics(train_model.summary)

    # tune pipeline before fit
    if not raw_data:
        print("Tuning binary pipeline...")
        tdf_cross, tdf_train = tune_binary(df, estimator)
        return tf_df, tdf_cross, tdf_train
    return tf_df

def tune_binary(df, estimator):
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")

    # validation split
    model, validationSplit_tdf = evaluate_with_train_validation_split(df, estimator, get_pipeline(df, estimator),
                                                                      evaluator)
    model.write().overwrite().save("models/binary/" + estimator + "/train_validation")

    # cross validation fit
    model, cross_valid_tdf = evaluate_with_cross_validation(df, estimator, get_pipeline(df, estimator), evaluator)
    model.write().overwrite().save("models/binary/" + estimator + "/cross_validation")

    print("The best models are saved in 'models/binary/*'.")

    return cross_valid_tdf, validationSplit_tdf


def prepare_stages_for_raw_data():
    # categorical_variables = ["frame.time_epoch", "frame.len", "eth.src", "eth.dst", "ip.src", "ip.dst", "tcp.srcport",
    #                          "tcp.dstport", "udp.srcport", "udp.dstport", "icmp.type", "icmp.code", "arp.opcode",
    #                          "arp.src.hw_mac", "arp.src.proto_ipv4", "arp.dst.hw_mac", "arp.dst.proto_ipv4", "ipv6.src",
    #                          "ipv6.dst"]
    categorical_variables = ["_c0", "_c1", "_c2", "_c3", "_c4", "_c5", "_c6",
                             "_c7", "_c8", "_c9", "_c10", "_c11", "_c12",
                             "_c13", "_c14", "_c15", "_c16", "_c17",
                             "_c18"]

    indexers = [StringIndexer(inputCol=column, outputCol=column + "-index", handleInvalid= "keep")
                for column in categorical_variables
                if column not in ["_c10", "_c11", "_c12", "_c13", "_c14", "_c15", "_c16", "_c17", "_c18"]]
    encoder = OneHotEncoder(
        inputCols=[indexer.getOutputCol() for indexer in indexers],
        outputCols=["{0}-encoded".format(indexer.getOutputCol()) for indexer in indexers]
    )
    assembler = VectorAssembler(
        inputCols=encoder.getOutputCols(),
        outputCol="features_v"
    )
    pca = PCA(k=15, inputCol="features_v", outputCol="features")
    return indexers + [encoder, assembler, pca]
