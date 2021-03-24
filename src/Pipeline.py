from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import dataframe, Window

# retrieve numeric data only
from pyspark.sql.functions import col, monotonically_increasing_id, lit
from pyspark.sql.functions import row_number

from src.DataLoader import show

k = 2


def numeric(df: dataframe.DataFrame):
    df.drop("").cache()  # fields with str value that are required to be dropped.


def generate_id(df: dataframe.DataFrame):
    # w = Window().orderBy(monotonically_increasing_id())
    df = df.withColumn("temp", lit('ABC'))
    w = Window().partitionBy('temp').orderBy(lit('A'))
    df = df.withColumn("id", row_number().over(w)).drop('temp')

    print("DataFrame with generated id: ")
    show(df)
    return df


def create_pipeline(df: dataframe.DataFrame):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features"
    )
    # assembled_df = assembler.transform(df)
    k_means = KMeans().setK(k).setSeed(1).setPredictionCol("cluster").setFeaturesCol("features")
    pipeline = Pipeline().setStages([assembler, k_means])
    # model = k_means.fit(assembled_df)
    pipeline_model = pipeline.fit(df)
    with_cluster = pipeline_model.transform(df)

    # print("Schema after transformation:")
    with_cluster.printSchema()

    # Shows the result.
    # transformed_df = with_cluster.select("cluster", "label").groupBy("cluster", "label").count()\
    #     .orderBy(col("cluster").asc(), col("label").desc())

    transformed_df_with_id = generate_id(with_cluster)

    # transformed_df_with_id.show()
    return transformed_df_with_id
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)

