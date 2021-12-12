### WITH PCA

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator decision_tree --mode binary > logs/pca/binary/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator decision_tree --mode multiclass > logs/pca/multiclass/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  comparative-analysis/main.py --estimator random_forest --mode binary > logs/pca/binary/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  comparative-analysis/main.py --estimator random_forest --mode multiclass > logs/pca/multiclass/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 --executor-memory 4G comparative-analysis/main.py --estimator perceptron --mode multiclass > logs/pca/multiclass/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator lr --mode binary > logs/pca/binary/lr/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator lr --mode multiclass > logs/pca/multiclass/lr/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator nb --mode binary > logs/pca/binary/nb/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator fm --mode binary > logs/pca/binary/fm/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator fm --mode multiclass > logs/pca/multiclass/fm/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

### WITHOUT PCA

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator decision_tree --mode binary > logs/without_pca/binary/dt/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator decision_tree --mode multiclass > logs/without_pca/multiclass/dt/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  comparative-analysis/main.py --estimator random_forest --mode binary > logs/without_pca/binary/rf/utput-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  comparative-analysis/main.py --estimator random_forest --mode multiclass > logs/without_pca/multiclass/rf/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator perceptron --mode multiclass > logs/without_pca/multiclass/perceptron/output-$(date +'%b-%d-%Y-%H:%M:%S').txt



### WITH CUSTOM EXECUTORS -- TODO

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 comparative-analysis/main.py --estimator random_forest --mode multiclass > logs/efficiency/output-exe15_core1-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3 --archives pyspark_venv.tar.gz  ws/comparative-analysis/main.py --estimator random_forest
#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3  --executor-memory 10GB --archives pyspark_venv.tar.gz  ws/comparative-analysis/main.py --estimator perceptron


## raw packet capture as input
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  comparative-analysis/main.py --estimator random_forest --mode binary --pcap true > logs/without_pca/binary/output_pcap-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  comparative-analysis/main.py --estimator random_forest --mode multiclass --pcap true > logs/without_pca/multiclass/output_pcap-$(date +'%b-%d-%Y-%H:%M:%S').txt
