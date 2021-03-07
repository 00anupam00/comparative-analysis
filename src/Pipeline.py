from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import dataframe

# retrieve numeric data only
k = 2


def numeric(df: dataframe.DataFrame):
    df.drop("").cache()  # fields with str value that are required to be dropped.


def create_pipeline(df: dataframe.DataFrame):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features"
    )
    assembled_df = assembler.transform(df)
    k_means = KMeans().setK(k).setSeed(1).setPredictionCol("prediction").setFeaturesCol("features")
    # pipeline = Pipeline().setStages([assembler, k_means])
    model = k_means.fit(assembled_df)
    # predictions = model.transform(df)
    # print("Schema after transformation:")
    # predictions.printSchema()
    # predictions.take(20)

    # evaluator = ClusteringEvaluator()
    # silhouette = evaluator.evaluate(predictions)
    # print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)
