# Import the needed libraries
import ast
import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark

# Set constants to point to the utility files and paths used
HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

# Set the corresponding environment variables
os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "/bin")
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

# Create the configuration
conf = SparkConf()
conf.set("spark.jars", JDBC_JAR)

# Create the spark session with the JDBC driver
spark = SparkSession.builder \
    .config(conf=conf) \
    .master("local") \
    .appName("testing") \
    .getOrCreate()

# Read the user credentials to access the database in PostgresFIB
with open('.uname_and_pwd', 'r') as file:
    credentials = ast.literal_eval(file.read())
    username = credentials['username']
    password = credentials['password']

# Connect to the database and read the needed table
aircraft_utilization = (spark.read
                        .format("jdbc")
                        .option("driver", "org.postgresql.Driver")
                        .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                        .option("dbtable", "public.aircraftutilization")
                        .option("user", username)
                        .option("password", password)
                        .load())

# Read the files in the resources/trainingData directory
sc = pyspark.SparkContext.getOrCreate()
csv_files = sc.wholeTextFiles('resources/trainingData/*')

# Obtain the mean of the sensor data per aircraft per day
# Join the aircraft_utilization rdd with the sensor data rdd
# Return the matrix (save it?)
