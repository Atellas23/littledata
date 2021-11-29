import ast
import os
import sys
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

with open('.uname_and_pwd', 'r') as file:
    credentials = ast.literal_eval(file.read())

username = credentials['username']
password = credentials['password']


aircraft_utilization = (spark.read
                        .format("jdbc")
                        .option("driver", "org.postgresql.Driver")
                        .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                        .option("dbtable", "public.aircraftutilization")
                        .option("user", username)
                        .option("password", password)
                        .load())

# print(aircraft_utilization.select('*').rdd.collect())
