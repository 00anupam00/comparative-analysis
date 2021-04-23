from pyspark.sql import dataframe

from src.utils.Utils import generate_id


def attach_labels(df_data_id, df_labels):
    # JOIN
    df = df_data_id.join(df_labels, on=["id"], how="inner")
    return df


def pre_process_data(df_data: dataframe, df_labels: dataframe):
    df_data_id = generate_id(df_data)
    df = attach_labels(df_data_id, df_labels)
    return df
