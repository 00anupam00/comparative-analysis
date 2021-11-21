from src.Paths import syn_dos_tsv, syn_dos_labels, ssl_reneg_tsv, ssl_reneg_labels, arp_spoof_labels, arp_spoof_tsv, \
    syn_dos_dataset, arp_spoof_pcap
from src.binary import BinaryPipeline, Evaluators
from src.binary.DataLoader import load_data
from src.featureextractor.FeatureExtractor import FE
from src.multiclass.DataPreProcessor import df_with_id, create_union, load_dataset_with_categories_for_reusability
from src.multiclass.Evaluators import evaluate_multiclass
from src.multiclass.MulticlassPipeline import process_multiclass_pipeline


def get_FE_instance(path):
    FE(path)

def run_reusability_test(mode, estimator):
    if mode == "binary":
        print("Binary Classification re-usability test... ")
        # load syn dos
        # get_FE_instance(syn_dos_pcapng)
        # df = load_data(syn_dos_tsv, syn_dos_labels, "true", "false")

        # load ssl_reneg dataset
        # get_FE_instance(ssl_reneg_pcapng)
        # df = load_data(ssl_reneg_tsv, ssl_reneg_labels, "true", "false")

        # load arp dataset
        get_FE_instance(arp_spoof_pcap)
        df = load_data(arp_spoof_tsv, arp_spoof_labels, "\t", "true", "false")
        df.na.drop("any")
        df = df.orderBy('id', ascending=False)

        tf_df = BinaryPipeline.process_binary_pipeline(df, estimator, True)

        # Evaluator
        print("\nEvaluating estimator for re-usability: ", str(estimator))
        print("Evaluation Results with default params: ")
        Evaluators.evaluate_binary_classifier(tf_df)
    elif mode == "multiclass":
        print("Multiclass Classification re-usability test... ")
        df = load_dataset_with_categories_for_reusability()
        df = df_with_id(df)
        tf_df, tdf_cross, tdf_train = process_multiclass_pipeline(df, estimator=estimator)
        print("\nEvaluating Estimator: ", str(estimator))
        print("Evaluation results with default params: ")
        evaluate_multiclass(tf_df)
