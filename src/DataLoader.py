from pcapfile import savefile
from pyspark.sql import SparkSession
from scapy.all import rdpcap

from src.Paths import pcap_filepath, labels_path

spark = SparkSession.builder.appName("outlier-detection").getOrCreate()


def load_data(path):
    df = spark.read.load(
        path,
        format="csv", sep=",", inferSchema="true", header="true")
    print("Loaded data with schema: ")
    return df


def show(df):
    df.printSchema()
    df.describe().show(25)


def analyze_labels(df):
    df = df.select("id", "label").groupBy("label").count().orderBy("count", ascending=False)
    show(df)


# Using pcapfile
def parse_pcap(filepath):
    test_cap = open(filepath, 'rb')
    pcap_file = savefile.load_savefile(test_cap, verbose=True)
    print("Pcap file ", pcap_file)


# Using scapy
def parse_pcap_with_scapy(filepath):
    a = rdpcap(filepath)

## TEST
if __name__ == '__main__':
    # analyze_labels(load_data(labels_path))
    parse_pcap(pcap_filepath) # not needed
