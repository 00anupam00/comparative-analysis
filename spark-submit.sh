
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator decision_tree > logs/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3 --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator decision_tree

spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  outlier-detection/main.py --estimator random_forest --mode binary> logs/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  outlier-detection/main.py --estimator random_forest --mode multiclass> logs/output-$(date +'%b-%d-%Y-%H:%M:%S').txt

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3 --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator random_forest

# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator perceptron

#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --conf spark.eventLog.enabled=false --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator perceptron

#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3  --executor-memory 10GB --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator perceptron
