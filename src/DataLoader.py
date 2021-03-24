from pyspark.sql import SparkSession, dataframe

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
data_path = "/Users/anupamrakshit/Documents/workspace/outlier-detection/input/SSL_Ren_first_2000_kfolds.csv"
# data_path = "/Users/anupamrakshit/Documents/workspace/dataset/OS_Scan_dataset.csv"
# label_path = "/Users/anupamrakshit/Documents/workspace/dataset/OS_Scan_labels.csv"


def load_data():
    df = spark.read.load(
        data_path,
        format="csv",
        sep=",",
        inferSchema="true",
        header="true")
    print("Loaded data with schema: ")
    # show(df)
    return df


def load_label():
    df_label = spark.read.load(
        label_path,
        format="csv", sep=",", inferSchema="true", header="false")
    # print("Loaded labels with indices")
    # show(df_label)


def with_labels(df: dataframe.DataFrame, df_labels=dataframe.DataFrame):
    df.join(

    )


def show(df):
    # df = df.select("label").groupBy("label").count().orderBy("count", ascending=False)
    df.printSchema()
    df.show(25)
