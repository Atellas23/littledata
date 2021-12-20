import ast
import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark
import datetime
import numpy as np

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "/bin")
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON


def parse_date(timestampstr: str) -> str:
    format = '%d%m%y'
    return str(datetime.datetime.strptime(timestampstr, format).date())


sc = pyspark.SparkContext.getOrCreate()
dd = sc.wholeTextFiles('resources/trainingData/*') \
    .map(lambda t: (t[0][-30:-4], t[1]))\
    .map(lambda t: ((parse_date(t[0][:6]), t[0][-6:]), t[1]))\
    .mapValues(lambda v: v.split('\n')[1:-2:])\
    .mapValues(lambda v: [float(r) for r in list(map(lambda element: element.split(';')[2], v))])\
    .reduceByKey(lambda a, b: a+b)\
    .mapValues(lambda v: np.mean(v))

with open('temp_results.txt', 'w') as f:
    print(dd.collect(), file=f)

# print(aircraft_utilization.select('*').rdd.collect())
