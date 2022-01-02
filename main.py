import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pipelines.dataManagement import process as data_management
from pipelines.dataAnalysis import process as data_analysis
from pipelines.runtimeClassifier import process as runtime_classifier

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

if(__name__ == "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # Create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    # Create and point to your pipelines here
    model_path = 'classification_tree.model'
    data_path = 'result.csv'
    data_management(sc, spark, data_path)
    data_analysis(spark, data_path, model_path)
    runtime_classifier(sc, spark, model_path)
