from pyspark.ml.feature import ChiSqSelector



def get_selector(featuresColName, labelColName):
    selector = ChiSqSelector(numTopFeatures=1, featuresCol=featuresColName,
                             outputCol="selectedFeatures", labelCol=labelColName)
    return selector
