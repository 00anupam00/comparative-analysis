import csv

from src import Paths
from src.Paths import arp_spoof_pcap, ssl_reneg_labels, arp_spoof_labels, syn_dos_labels, arp_vec_path, syn_vec_path, \
    ssl_vec_path
from src.binary import BinaryPipeline, Evaluators
from src.binary.DataLoader import load_data
from src.multiclass.MulticlassDataLoader import load_data
from src.featureextractor.FeatureExtractor import FE
from src.multiclass.DataPreProcessor import create_union
from src.multiclass.Evaluators import evaluate_multiclass
from src.multiclass.MulticlassPipeline import process_multiclass_pipeline


def pcap_feature_preparation(mode, estimator):
    if mode == "binary":
        print("Binary Classification re-usability test... ")
        # load syn dos
        # get_FE_instance(syn_dos_pcapng)
        # df = load_data(syn_dos_tsv, syn_dos_labels, "true", "false")

        # load ssl_reneg dataset
        # get_FE_instance(ssl_reneg_pcapng)
        # df = load_data(ssl_reneg_tsv, ssl_reneg_labels, "true", "false")

        # load arp dataset
        fe = get_FE_instance(arp_spoof_pcap)
        convert_write_to_csv(arp_vec_path, fe)
        df = load_data(arp_vec_path, Paths.arp_spoof_labels)
        df = df.orderBy('id', ascending=False)

        # feed dataset to binary pipeline
        tf_df = BinaryPipeline.process_binary_pipeline(df, estimator)

        # Evaluator
        print("\nEvaluating estimator for re-usability: ", str(estimator))
        print("Evaluation Results with default params: ")
        Evaluators.evaluate_binary_classifier(tf_df)
    elif mode == "multiclass":
        print("Multiclass Classification re-usability test... ")

        df = prepare_dataset()

        # feed dataset to multiclass pipeline
        tf_df, tdf_cross, tdf_train = process_multiclass_pipeline(df, estimator=estimator)
        print("\nEvaluating Estimator: ", str(estimator))
        print("Evaluation results with default params: ")
        evaluate_multiclass(tf_df)


def get_FE_instance(path):
    return FE(path)

## binary
def convert_write_to_csv(path, fe):
    vecs = []
    while True:
        v = fe.get_next_vector()
        if (len(v) == 0):
            break
        vecs.append(v)
    with open(path, 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(vecs)


## multiclass
def prepare_dataset():
    # create arp-attack feature vectors
    fe = get_FE_instance(arp_spoof_pcap)
    convert_write_to_csv(arp_vec_path, fe)

    # create syn-attack feature vectors
    fe = get_FE_instance(Paths.syn_dos_pcap)
    convert_write_to_csv(syn_vec_path, fe)

    # create ssl-attack feature vectors
    fe = get_FE_instance(Paths.ssl_reneg_pcap)
    convert_write_to_csv(ssl_vec_path, fe)

    # laod data
    df_ssl = load_data(ssl_vec_path, ssl_reneg_labels, 1)
    df_arp = load_data(arp_vec_path, arp_spoof_labels, 2)
    df_syn = load_data(syn_vec_path, syn_dos_labels, 3)

    return create_union([df_ssl, df_arp, df_syn])