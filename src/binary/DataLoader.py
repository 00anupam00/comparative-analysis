from pyspark.sql.types import StructType, StructField, LongType, IntegerType

from src.SparkConfig import get_spark_session
from src.binary.PrePocessor import pre_process_data

spark = get_spark_session("outlier-detection")


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
        mode="DROPMALFORMED",
        header="true")
    print("Loaded labels. ")

    # Pass through data pre-processor
    df = pre_process_data(df_data, df_labels)
    # analyze_labels(df)
    return df


def show(df):
    df.describe().show(25)


def analyze_labels(df):
    df = df.select("id", "label").groupBy("label").count().orderBy("count", ascending=False)
    show(df)
