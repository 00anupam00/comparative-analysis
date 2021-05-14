start-master.sh
start-worker.sh spark://rakshit-thesis.cloud.ut.ee:7077
pip install -r outlier-detection/requirements.txt
venv-pack -o pyspark_venv.tar.gz
