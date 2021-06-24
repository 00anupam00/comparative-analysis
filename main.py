import getopt
import sys

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from src.Paths import syn_dos_labels, \
    syn_dos_dataset
# from src.MulticlassPipeline import process_multiclass_pipeline
from src.Tuning import evaluate_with_train_validation_split, evaluate_with_cross_validation, get_pipeline
from src.binary import Evaluators, BinaryPipeline
from src.binary.DataLoader import load_data
from src.multiclass.DataPreProcessor import load_dataset_with_categories, df_with_id
from src.multiclass.Evaluators import evaluate_multiclass
from src.multiclass.MulticlassPipeline import process_multiclass_pipeline
from src.utils import Estimators

def binaryClassify(estimator):
    # 1. Load and pre-process data
    # df = DataLoader.load_data(ssl_reneg_dataset, ssl_reneg_labels)
    # df = DataLoader.load_data(arp_spoof_dataset, arp_spoof_labels)
    df = load_data(syn_dos_dataset, syn_dos_labels)
    # fixme 1.1 Limit the last 100000 records for preserving memory
    df = df.orderBy('id', ascending=False).limit(100000)

    # tf_df = Pipeline.create_pipeline(df)
    tf_df = BinaryPipeline.create_pipeline(df, estimator)

    # Evaluator
    Evaluators.evaluate_binary_classifier(tf_df)

    # FIXME
    # Visualize
    tf_df.printSchema()
    # Visualization.visualize_data(tf_df)


def multiclassClassify(estimator):
    df = load_dataset_with_categories()
    df = df_with_id(df)
    tf_df = process_multiclass_pipeline(df, estimator=estimator)
    print("Metrics for Estimator: ", str(estimator))
    evaluate_multiclass(tf_df)


def tune_multiclass(estimator):
    df = load_dataset_with_categories()
    print("Tuning model: ", str(estimator))
    evaluate_with_cross_validation(df, get_pipeline(df, estimator), estimator, MulticlassClassificationEvaluator())
    evaluate_with_train_validation_split(df, get_pipeline(df, estimator), estimator, MulticlassClassificationEvaluator())


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
            estimator = arg.strip()

    print("Selected Estimator is: ", estimator)

    binaryClassify(estimator=estimator)  # todo uncomment for binary classifiers
    # multiclassClassify(estimator=estimator)

    # TUNE
    # tune_multiclass(estimator)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(sys.argv[1:])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
