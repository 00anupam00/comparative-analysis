from pyspark.sql import dataframe
import matplotlib.pyplot as plt


def convert_to_pandas(df_predicted: dataframe.DataFrame):
    pds_df = df_predicted.toPandas().set_index('id')
    pds_df.head()
    return pds_df


def visualize_data(tf_df: dataframe.DataFrame):
    pds_df = convert_to_pandas(tf_df)
    three_d = plt.figure(figsize=(12, 10)).gca(projection='3d')
    three_d.scatter(pds_df.x, pds_df.y, pds_df.z, c=pds_df.prediction)
    three_d.set_xlabel('x')
    three_d.set_ylabel('y')
    three_d.set_zlabel('z')
    plt.show()
    # use spark dataframe.plot
