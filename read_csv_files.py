import ast
import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "/bin")
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

sc = pyspark.SparkContext.getOrCreate()
sc.wholeTextFiles('resources/trainingData/*')

# print(aircraft_utilization.select('*').rdd.collect())
