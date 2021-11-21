import os


basePath = os.path.abspath(os.getcwd())

ssl_reneg_dataset = basePath + "/input/ssl/dataset.csv"
ssl_reneg_pcap = basePath + "/input/ssl/dataset.pcap"
ssl_reneg_tsv = basePath + "/input/ssl/dataset.pcap.tsv"
ssl_reneg_labels = basePath + "/input/ssl/labels.csv"

arp_spoof_dataset = basePath + "/input/arp/dataset.csv"
arp_spoof_pcap = basePath + "/input/arp/dataset.pcap"
arp_spoof_tsv = basePath + "/input/arp/dataset.pcap.tsv"
arp_spoof_labels = basePath + "/input/arp/labels.csv"

syn_dos_dataset = basePath + "/input/syn/dataset.csv"
syn_dos_pcap = basePath + "/input/syn/dataset.pcap"
syn_dos_tsv = basePath + "/input/syn/dataset.pcap.tsv"
syn_dos_labels = basePath + "/input/syn/labels.csv"

