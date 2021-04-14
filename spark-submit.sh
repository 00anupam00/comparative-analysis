./bin/spark-submit \
--deploy-mode cluster \
--master https://kubernetes.docker.internal:6443 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark-sa \
--name spark-pi \
--class org.apache.spark.examples.SparkPi \
--conf spark.executor.instances=2  \
--conf spark.kubernetes.executor.container.image= 00anupam00/spark-py \
local:///Users/anupamrakshit/Documents/workspace/outlier-detection/src/SimpleApp.py