from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import dataframe
from pyspark.sql.functions import col


# get an estimate of K using elbow method
def determine_K():
    return 3


def process_pipeline(df: dataframe.DataFrame):
    assembler = VectorAssembler(
        inputCols=[x for x in df.columns if x != "label"],
        outputCol="features"
    )
    # assembled_df = assembler.transform(df)
    k_means = KMeans().setK(determine_K()).setSeed(1).setPredictionCol("cluster").setFeaturesCol("features")
    pipeline = Pipeline().setStages([assembler, k_means])
    # model = k_means.fit(assembled_df)
    pipeline_model = pipeline.fit(df)
    transformed_df = pipeline_model.transform(df)

    # print("Schema after transformation:")
    transformed_df.printSchema()

    # Shows the result.
    # transformed_df = with_cluster.select("cluster", "label").groupBy("cluster", "label").count() \
    #     .orderBy(col("cluster").asc(), col("label").desc())

    print("Transformed df:")
    transformed_df.show()
    return transformed_df
