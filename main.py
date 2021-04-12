import sys, getopt

from src import DataLoader, BinaryPipeline, Estimators, Evaluators
from src.Paths import ssl_reneg_dataset, ssl_reneg_labels


def run(argv):
    estimator = ''
    try:
        opts, args = getopt.getopt(argv, "he:", ["estimator="])
    except getopt.GetoptError:
        print('main.py -e <estimators>')
        print('Estimators value could be one of: ', str(Estimators.get_estimator_keys()))
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -e <estimators>')
            print('Estimators value could be one of: ', str(Estimators.get_estimator_keys()))
        elif opt in ['-e', '--estimator']:
            estimator = arg

    print("Selected Estimator is: ", estimator)

    # 1. Load and pre-process data
    df = DataLoader.load_data(ssl_reneg_dataset, ssl_reneg_labels)

    # fixme 1.1 Limit the last 100000 records for preserving memory
    df = df.orderBy('id', ascending=False).limit(100000)

    #2. Create Pipelines
    # tf_df = Pipeline.create_pipeline(df)
    tf_df = BinaryPipeline.create_pipeline(df, estimator)

    #3. Evaluate
    # Evaluator
    Evaluators.evaluate_binary_classifier(tf_df)

    # FIXME
    # Visualize
    # Visualization.visualize_data(tf_df)

    # todo
    # Fix the dataset with appropriate labels. Or use another dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(sys.argv[1:])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
