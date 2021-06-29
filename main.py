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
    print("Binary Classification... ")
    df = load_data(syn_dos_dataset, syn_dos_labels)
    # fixme 1.1 Limit the last 100000 records for preserving memory
    df = df.orderBy('id', ascending=False).limit(100000)

    # tf_df = Pipeline.create_pipeline(df)
    tf_df, tdf_cross, tdf_train = BinaryPipeline.process_binary_pipeline(df, estimator)

    # Evaluator
    print("Metrics for estimator: ", str(estimator))
    print("Metrics for default params: ")
    Evaluators.evaluate_binary_classifier(tf_df)

    print("Metrics for hyper params tuned with cross validation: ")
    Evaluators.evaluate_binary_classifier(tdf_cross)
    print("Metrics for hyper params tuned with train validation split: ")
    Evaluators.evaluate_binary_classifier(tdf_train)

    # FIXME
    # Visualize
    tf_df.printSchema()
    # Visualization.visualize_data(tf_df)


def multiclassClassify(estimator):
    print("Multiclass Classification... ")
    df = load_dataset_with_categories()
    df = df_with_id(df)
    tf_df, tdf_cross, tdf_train = process_multiclass_pipeline(df, estimator=estimator)
    print("Metrics for Estimator: ", str(estimator))
    print("Metrics for default params: ")
    evaluate_multiclass(tf_df)
    print("Metrics for hyper params tuned with cross validation: ")
    evaluate_multiclass(tdf_cross)
    print("Metrics for hyper params tuned with train validation split: ")
    evaluate_multiclass(tdf_train)




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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(sys.argv[1:])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
