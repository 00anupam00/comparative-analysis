from pyspark.sql.functions import regexp_replace, when, lit
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType

from src.SparkConfig import get_spark_session
from src.binary.PrePocessor import pre_process_data

spark = get_spark_session("outlier-detection")


def load_data(data_path, labels_path, multiclass_param):
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
    df_labels = df_labels.withColumn('label', when(df_labels.label == 1, lit(multiclass_param)).otherwise(lit(0)))

    print("Loaded labels. ")
    # df_labels.select("id", "label").orderBy('id', ascending=False).show()

    # Pass through data pre-processor
    df = pre_process_data(df_data, df_labels)
    return df


def show(df):
    df.describe().show(25)
    print("The last 10 lines of the dataset: ")
    print(df.tail(10))