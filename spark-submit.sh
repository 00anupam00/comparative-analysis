#spark-submit --deploy-mode cluster --master k8s://https://127.0.0.1:16443 --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark-sa --name spark-pi --class pi.py --conf spark.executor.instances=2  --conf spark.kubernetes.executor.container.image=00anupam00/spark-py /home/ubuntu/spark/spark-3.1.1-bin-hadoop2.7/examples/src/main/python/pi

# spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 outlier-detection/main.py --estimator decision_tree
#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3 --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator decision_tree
spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077  outlier-detection/main.py --estimator random_forest > logs/output-$(date +'%b-%d-%Y-%H:%M:%S').txt
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3 --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator random_forest
# spark-submit  --master spark://rakshit-thesisv2.cloud.ut.ee:7077 outlier-detection/main.py --estimator perceptron
#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --conf spark.eventLog.enabled=false --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator perceptron

#spark-submit  --master spark://rakshit-thesis.cloud.ut.ee:7077 --num-executors 5 --executor-cores 3  --executor-memory 10GB --archives pyspark_venv.tar.gz  ws/outlier-detection/main.py --estimator perceptron
