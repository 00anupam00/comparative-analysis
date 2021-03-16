from src import DataLoader, Pipeline, Visualization


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = DataLoader.load_data()
    tf_df = Pipeline.create_pipeline(df)

    # FIXME
    Visualization.visualize_data(tf_df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
