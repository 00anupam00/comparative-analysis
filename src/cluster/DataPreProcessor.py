from pyspark.sql.functions import lit
from pyspark.sql.types import StructType

from src.SparkConfig import get_spark_session
from src.Paths import ssl_reneg_dataset, arp_spoof_dataset, syn_dos_dataset
from src.utils.Utils import generate_id

spark = get_spark_session("outlier-detection")


def load_dataset_with_categories():
    df_ssl = spark.read.load(
        ssl_reneg_dataset,
        format="csv", sep=",", inferSchema="true", header="false").withColumn("category", lit(0))
    df_arp = spark.read.load(
        arp_spoof_dataset,
        format="csv", sep=",", inferSchema="true", header="false").withColumn("category", lit(1))
    df_syn = spark.read.load(
        syn_dos_dataset,
        format="csv", sep=",", inferSchema="true", header="false").withColumn("category", lit(2))
    print("Loaded dataset. ")

    # FIXME
    df_arp, df_ssl, df_syn = limit_rows(df_arp, df_ssl, df_syn)

    df = create_union([df_ssl, df_syn, df_arp])
    return df


# limit the dataset for preserving memory
def limit_rows(df_arp, df_ssl, df_syn):
    df_ssl = df_ssl.limit(10000)
    df_arp = df_arp.limit(10000)
    df_syn = df_syn.limit(10000)
    return df_arp, df_ssl, df_syn


def df_with_id(df):
    return generate_id(df)


def create_union(dfs: list):
    schema = dfs[0].schema
    df_U = spark.createDataFrame([], schema)
    for df in dfs:
        df_U.union(df)
    return df_U


# Test
if __name__ == '__main__':
    df = load_dataset_with_categories()
    df = df_with_id(df)
    df.printSchema()
    df.count()
