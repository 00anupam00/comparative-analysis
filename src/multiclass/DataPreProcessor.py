from pyspark.sql.types import StringType

from src.Paths import ssl_reneg_dataset, arp_spoof_dataset, syn_dos_dataset, ssl_reneg_labels, arp_spoof_labels, \
    syn_dos_labels
from src.SparkConfig import get_spark_session
from src.utils.MulticlassDataLoader import load_data
from src.utils.Utils import generate_id

spark = get_spark_session("outlier-detection")


def load_dataset_with_categories():
    df_ssl = load_data(ssl_reneg_dataset, ssl_reneg_labels, 1)
    df_arp = load_data(arp_spoof_dataset, arp_spoof_labels, 2)
    df_syn = load_data(syn_dos_dataset, syn_dos_labels, 3)
    # df_syn = spark.createDataFrame([], StringType())

    df = create_union([df_ssl, df_arp, df_syn])
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
        df_U = df_U.union(df)
    return df_U
