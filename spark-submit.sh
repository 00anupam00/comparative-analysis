### WITH PCA

 spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator decision_tree --mode binary > logs/pca/binary/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator decision_tree --mode multiclass > logs/pca/multiclass/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  outlier-detection/main.py --estimator random_forest --mode binary > logs/pca/binary/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  outlier-detection/main.py --estimator random_forest --mode multiclass > logs/pca/multiclass/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator perceptron





### WITHOUT PCA

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator decision_tree --mode binary> logs/without_pca/binary/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator decision_tree --mode multiclass> logs/without_pca/multiclass/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  outlier-detection/main.py --estimator random_forest --mode binary> logs/without_pca/binary/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  outlier-detection/main.py --estimator random_forest --mode multiclass> logs/without_pca/multiclass/output-$(date +'%b-%d-%Y-%H:%M:%S').txt



### WITH CUSTOM EXECUTORS -- TODO

#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3 --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator decision_tree
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3 --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator random_forest
#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3  --executor-memory 10GB --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator perceptron