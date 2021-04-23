# Evaluate clustering by computing Silhouette score
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeansModel
from pyspark.sql import dataframe


# Ref: https://www.bmc.com/blogs/python-spark-k-means-example/
def evaluate_KMeans(predictions : dataframe.DataFrame):
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    return silhouette


def compute_cost(model: KMeansModel, dataset: dataframe.DataFrame):
    # Evaluate clustering.
    cost = model.computeCost(dataset)
    print("Within Set Sum of Squared Errors = " + str(cost))


# Shows the result.
def kMeans_result(model: KMeansModel):
    print("Cluster Centers: ")
    ctr=[]
    centers = model.clusterCenters()
    for center in centers:
        ctr.append(center)
        print(center)
