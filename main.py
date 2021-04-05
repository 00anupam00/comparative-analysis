import sys, getopt

from src import DataLoader, BinaryPipeline, Estimators, Evaluators


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

    df = DataLoader.load_data()
    # tf_df = Pipeline.create_pipeline(df)
    tf_df = BinaryPipeline.create_pipeline(df, estimator)

    # Evaluator
    Evaluators.evaluate_binary_classifier(tf_df)

    # FIXME
    # Visualization.visualize_data(tf_df)

    # todo
    # Fix the dataset with appropriate labels. Or use another dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(sys.argv[1:])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
