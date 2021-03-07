from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
data_path = "/Users/anupamrakshit/Documents/workspace/outlier-detection/input/SSL_Ren_first_2000_kfolds.csv"


def load_data():
    df = spark.read.load(
        data_path,
        format="csv", sep=",", inferSchema="true", header="true")
    print("Loaded data with schema: ")
    return df


def show(df):
    df = df.select("label").groupBy("label").count().orderBy("count", ascending=False)
    df.printSchema()
    df.describe().show(25)
