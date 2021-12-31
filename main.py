import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pipelines.dataManagement import process as data_management
from pipelines.dataAnalysis import process as data_analysis
from pipelines.runtimeClassifier import process as runtime_classifier
from test import process as process_test

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

if(__name__ == "__main__"):
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} (dm, da, rc)')
        exit()
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    # data_filepath = data_management(sc, spark)
    # model_path = data_analysis(sc, spark, data_filepath)

    # Create and point to your pipelines here
    if sys.argv[1] == 'dm':
        # 1. Data Management Pipeline
        data_filepath = data_management(sc, spark)
        print('Starting data management pipeline...')
        print(data_filepath)
    elif sys.argv[1] == 'da':
        print('Starting data analysis pipeline...')
        model_path = data_analysis(sc, spark, 'result.csv')
        print(model_path)
    elif sys.argv[1] == 'rc':
        print('Starting runtime classifier pipeline...')
        res = runtime_classifier(sc, spark, 'classification_tree.model')
        print('Finished runtime classifier pipeline')
    else:
        print('error: wrong option!')
        raise NotImplementedError()
