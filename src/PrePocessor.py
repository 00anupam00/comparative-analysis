from pyspark.sql import dataframe
from pyspark.sql.functions import monotonically_increasing_id, shiftRight, when, lit, expr, broadcast, sum
from pyspark.sql.window import Window


def generate_id(df_data):
    # Generate monotonous id
    df_data = df_data.withColumn("row_id", monotonically_increasing_id())  # starts with 0

    # analyze spark internal partitions
    df_partition = df_data.withColumn("partition_id", shiftRight('row_id', 33)) \
        .withColumn("row_offset", df_data['row_id'].bitwiseAND(2147483647))
    partitions_size = df_partition.groupBy("partition_id").count().withColumnRenamed("count", "partition_size")
    windowSpec = Window.orderBy("partition_id").rowsBetween(Window.unboundedPreceding, -1) # warnings for window without a partition

    # Take care of the null partition_offsets
    partitions_offset = partitions_size.withColumn("partition_offset", when(expr("partition_id = 0"), lit(0))
                                                   .otherwise(sum("partition_size").over(windowSpec)))
    df_data_id = df_partition.join(broadcast(partitions_offset), "partition_id").withColumn("id",partitions_offset.partition_offset + df_partition.row_offset + 1).drop("partition_id", "row_id", "row_offset", "partition_size", "partition_offset")
    df_data_id.select('id', '_c0', '_c114').orderBy('id', ascending=False).show()  ## Check the data
    return df_data_id


def attach_labels(df_data_id, df_labels):
    # JOIN
    df = df_data_id.join(df_labels, on=["id"], how="inner")
    return df


def pre_process_data(df_data: dataframe, df_labels: dataframe):
    df_data_id = generate_id(df_data)
    df = attach_labels(df_data_id, df_labels)
    return df
