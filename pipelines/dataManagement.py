# Import the needed libraries
import ast
import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark
import numpy as np
from functools import reduce
from datetime import datetime as dt
import datetime


# # Set constants to point to the utility files and paths used
# HADOOP_HOME = "./resources/hadoop_home"
# JDBC_JAR = "./resources/postgresql-42.2.8.jar"
# PYSPARK_PYTHON = "python3.8"
# PYSPARK_DRIVER_PYTHON = "python3.8"

# # Set the corresponding environment variables
# os.environ["HADOOP_HOME"] = HADOOP_HOME
# sys.path.append(HADOOP_HOME + "/bin")
# os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
# os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

# # Create the configuration
# conf = SparkConf()
# conf.set("spark.jars", JDBC_JAR)

# # Create the spark session with the JDBC driver
# spark = SparkSession.builder \
#     .config(conf=conf) \
#     .master("local") \
#     .appName("testing") \
#     .getOrCreate()


def process(spark_context: pyspark.SparkContext, spark_session: SparkSession):
    # Read the user credentials to access the database in PostgresFIB
    with open('.uname_and_pwd', 'r') as file:
        credentials = ast.literal_eval(file.read())
        username = credentials['username']
        password = credentials['password']

    # Connect to the database and read the needed table as an RDD
    aircraft_utilization_rdd = (spark_session.read
                                .format("jdbc")
                                .option("driver", "org.postgresql.Driver")
                                .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                                .option("dbtable", "public.aircraftutilization")
                                .option("user", username)
                                .option("password", password)
                                .load()
                                .select('aircraftid', 'timeid', 'flighthours', 'flightcycles', 'delayedminutes')
                                .rdd
                                .map(lambda t: ((str(t[1]), t[0]), t[2::]))
                                .mapValues(lambda v: (float(v[0]), int(v[1]), int(v[2])))
                                )

    # Define a function to parse out the date from a string
    def parse_date(timestampstr: str, format: str = '%Y-%m-%d') -> str:
        return str(dt.strptime(timestampstr, format).date())

    operation_interruption_rdd = (spark_session.read
                                  .format("jdbc")
                                  .option("driver", "org.postgresql.Driver")
                                  .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
                                  .option("dbtable", "oldinstance.operationinterruption")
                                  .option("user", username)
                                  .option("password", password)
                                  .load()
                                  .select('aircraftregistration', 'starttime', 'kind')
                                  .rdd
                                  .map(lambda t: ((parse_date(str(t[1])[:19], '%Y-%m-%d %H:%M:%S'), str(t[0])), 1, t[-1]))
                                  .filter(lambda t: t[-1] in ('Delay', 'Safety', 'AircraftOnGround'))
                                  .map(lambda t: t[:-1])
                                  )

    # Read the files in the resources/trainingData directory and
    # obtain the mean of the sensor data per aircraft per day
    # sc = pyspark.SparkContext.getOrCreate()
    csv_files = spark_context.wholeTextFiles('resources/trainingData/*')\
        .map(lambda t: (t[0][-30:-4], t[1]))\
        .map(lambda t: ((parse_date(t[0][:6], '%d%m%y'), t[0][-6:]), t[1]))\
        .mapValues(lambda v: v.split('\n')[1:-2:])\
        .mapValues(lambda v: [(float(r), 1) for r in list(map(lambda element: element.split(';')[2], v))])\
        .mapValues(lambda v: tuple(reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]), v)))\
        .mapValues(lambda v: (v[0]/v[1], v[1]))\
        .reduceByKey(lambda a, b: ((a[0]*a[1]+b[0]*b[1])/(a[1]+b[1]), a[1]+b[1]))\
        .mapValues(lambda v: v[0])

    # Join the aircraft_utilization rdd with the sensor data rdd
    data = aircraft_utilization_rdd.join(
        csv_files).mapValues(lambda v: (*v[0], v[1]))

    # Aquest tros no funciona
    ##

    def any_maintenances_in_post_7_days(aircraftid: str, timeid: str, format: str = '%Y-%m-%d'):
        for d in range(7):
            if operation_interruption_rdd.filter(lambda t: t[0][0] == str(dt.strptime(timeid, format).date()+datetime.timedelta(days=d)) and t[0][1] == aircraftid).count() > 0:
                return True
    # Bàsicament, el problema és que no es poden fer transformacions dins de transformacions
    # (filter inside of map)
    data = data.map(lambda t: (
        t[0], any_maintenances_in_post_7_days(t[0][1], t[0][0]), *t[1]))
    ##
    print(data.first())
    return None

    # Return the matrix (save it?)
    with open('result.csv', 'w') as f:
        data = data.values().collect()
        result = 'FH,FC,DM,SensorAVG,Count\n'
        for v in data:
            aux = map(str, v)
            result += ','.join(aux)+'\n'
        print(result, file=f)

    return 'result.csv'
