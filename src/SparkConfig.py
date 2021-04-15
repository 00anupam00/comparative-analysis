from pyspark.sql import SparkSession


def get_spark_session(appName):
    return SparkSession.builder.appName(appName).getOrCreate()