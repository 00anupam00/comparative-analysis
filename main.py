import getopt
import sys
from datetime import datetime

from src.Paths import syn_dos_labels, \
    syn_dos_dataset
from src.binary import Evaluators, BinaryPipeline
from src.binary.DataLoader import load_data
from src.multiclass.DataPreProcessor import load_dataset_with_categories, df_with_id
from src.multiclass.Evaluators import evaluate_multiclass
from src.multiclass.MulticlassPipeline import process_multiclass_pipeline
from src.featureextractor.PcapFeaturePreparation import pcap_feature_preparation
from src.utils import Estimators


def binaryClassify(estimator):
    print("Binary Classification... ")
    # load syn dos
    # df = load_data(syn_dos_dataset, syn_dos_labels)

    # load ssl_reneg dataset
    # df = load_data(ssl_reneg_dataset, ssl_reneg_labels)
    #
    # load arp dataset
    # df = load_data(arp_spoof_dataset, arp_spoof_labels)

    df = df.orderBy('id', ascending=False)

    tf_df, tdf_cross, tdf_train = BinaryPipeline.process_binary_pipeline(df, estimator)

    # Evaluator
    print("\nEvaluating estimator: ", str(estimator))
    print("Evaluation results with default params: ")
    Evaluators.evaluate_binary_classifier(tf_df)

    print("\nEvaluation results for hyper params tuned with cross validation: ")
    Evaluators.evaluate_binary_classifier(tdf_cross)
    print("\nEvaluation results for hyper params tuned with train validation split: ")
    Evaluators.evaluate_binary_classifier(tdf_train)


def multiclassClassify(estimator):
    print("Multiclass Classification... ")
    df = load_dataset_with_categories()
    df = df_with_id(df)
    tf_df, tdf_cross, tdf_train = process_multiclass_pipeline(df, estimator=estimator)
    print("\nEvaluating Estimator: ", str(estimator))
    print("Evaluation results with default params: ")
    evaluate_multiclass(tf_df)
    print("\nEvaluation results for hyper params tuned with cross validation: ")
    evaluate_multiclass(tdf_cross)
    print("\nEvaluation results for hyper params tuned with train validation split: ")
    evaluate_multiclass(tdf_train)


def run(argv):
    startTime = datetime.now()
    print("Application started at : ", startTime)
    estimator = ''
    mode = ''
    raw = False
    try:
        opts, args = getopt.getopt(argv, "he:m:p:", ["estimator=", "mode=", "pcap="])
    except getopt.GetoptError:
        print('main.py -e <estimators> -m <binary|multiclass> -r <true|false>')
        print('Estimators value could be one of: ', str(Estimators.get_estimator_keys()))
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -e <estimators>')
            print('Estimators value could be one of: ', str(Estimators.get_estimator_keys()))
        elif opt in ['-e', '--estimator']:
            estimator = arg.strip()
        elif opt in ['-m', '--mode']:
            mode = arg.strip()
        elif opt in ['-r', '--pcap']:
            if arg.strip().lower() in ['true', 'y', 'yes']:
                raw = True
    print("Running application with the following arguments:")
    print("Estimator: ", str(estimator))
    print("Mode: ", str(mode))
    print("Pcap File: ", str(reuse))


    print("Selected Estimator is: ", estimator)

    if raw:
        print("Running re-usability test.")
        pcap_feature_preparation(mode, estimator)
    else:
        if "binary" == mode:
            binaryClassify(estimator=estimator)
        elif "multiclass" == mode:
            multiclassClassify(estimator=estimator)

    endTime = datetime.now()
    print("Application finished at : ", endTime)
    print("Total time taken to process: ", str(endTime - startTime))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(sys.argv[1:])
