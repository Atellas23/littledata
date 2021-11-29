import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "/bin")
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

conf = SparkConf()  # create the configuration
conf.set("spark.jars", JDBC_JAR)

spark = SparkSession.builder \
    .config(conf=conf) \
    .master("local") \
    .appName("testing") \
    .getOrCreate()

username = 'alex.batlle'
password = 'DB050901'


aircraft_utilization = (spark.read
                        .format("jdbc")
                        .option("driver", "org.postgresql.Driver")
                        .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                        .option("dbtable", "public.aircraftutilization")
                        .option("user", username)
                        .option("password", password)
                        .load())

print(aircraft_utilization.select('*').rdd.collect())
