from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType

from src.Paths import ssl_reneg_labels, ssl_reneg_dataset
from src.PrePocessor import pre_process_data

spark = SparkSession.builder.appName("outlier-detection").getOrCreate()


def load_data(data_path, labels_path):
    df_data = spark.read.load(
        data_path,
        format="csv", sep=",", inferSchema="true", header="false")
    print("Loaded dataset. ")

    labels_schema = StructType([StructField("id", LongType(), False),
                                StructField("label", IntegerType(), False)])
    df_labels = spark.read.load(
        labels_path,
        format="csv",
        sep=",",
        inferSchema="false",
        schema=labels_schema,
        header="true")
    print("Loaded labels. ")

    # Pass through data pre-processor
    df = pre_process_data(df_data, df_labels)
    return df


def show(df):
    df.describe().show(25)
    print("The last 10 lines of the dataset: ")
    print(df.tail(10))


def analyze_labels(df):
    df = df.select("id", "label").groupBy("label").count().orderBy("count", ascending=False)
    show(df)

