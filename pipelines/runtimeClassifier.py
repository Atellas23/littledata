# Import the needed libraries
import ast
from pyspark.sql import SparkSession
import pyspark
from functools import reduce
from datetime import datetime as dt
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, IndexToString
import sys
import os
from pyspark import SparkConf


def process(spark_context: pyspark.SparkContext, spark_session: SparkSession, model_path: str):
    # Read the AircraftID and TimeID from command line
    aircraft_id = input('Insert name of aircraft: ')
    time_id = input('Insert day (ddmmyy): ')
    # Read the user credentials to access the database in PostgresFIB
    # This is currently read from a text file formatted like a python
    # dictionary, with 'username' and 'password' fields
    with open('.uname_and_pwd', 'r') as file:
        credentials = ast.literal_eval(file.read())
        username = credentials['username']
        password = credentials['password']

    # Define a function to parse out the date from a string
    def parse_date(timestampstr: str, format: str = '%Y-%m-%d') -> str:
        return str(dt.strptime(timestampstr, format).date())

    # Connect to the data warehouse and read the needed table as an RDD
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
                                .filter(lambda t: t[0] == aircraft_id and parse_date(str(t[1])) == parse_date(time_id, '%d%m%y'))
                                .map(lambda t: ((str(t[1]), t[0]), t[2::]))
                                .mapValues(lambda v: (float(v[0]), int(v[1]), int(v[2])))
                                )

    # Failsafe: if the aircraft_utilization_rdd object is empty, it means that
    # there is no tuple in the data warehouse corresponding to that day and aircraft
    if aircraft_utilization_rdd.isEmpty():
        print(
            f'ERROR: there is no such pair (aircraftId, timeId) as ({aircraft_id}, {time_id}) in the data warehouse')
        print('exiting...')
        return

    # Read the files in the resources/trainingData directory and
    # obtain the mean of the sensor data per aircraft per day
    # The path of the files should be changed to where the files
    # are located in your system
    sensor_data = (spark_context.wholeTextFiles('resources/trainingData/*')
                   .map(lambda t: (t[0][-30:-4], t[1]))
                   .filter(lambda t: t[0][:6] == time_id and t[0][-6:] == aircraft_id)
                   .map(lambda t: ((parse_date(t[0][:6], '%d%m%y'), t[0][-6:]), t[1]))
                   .mapValues(lambda v: v.split('\n')[1:-2:])
                   .mapValues(lambda v: [(float(r), 1) for r in list(map(lambda element: element.split(';')[2], v))])
                   .mapValues(lambda v: tuple(reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]), v)))
                   .mapValues(lambda v: (v[0]/v[1], v[1]))
                   .reduceByKey(lambda a, b: ((a[0]*a[1]+b[0]*b[1])/(a[1]+b[1]), a[1]+b[1]))
                   .mapValues(lambda v: v[0])
                   )

    # Failsafe: if the sensor_data object is empty, it means that there is no
    # tuple in the sensor data files corresponding to that day and aircraft
    if sensor_data.isEmpty():
        print(
            f'ERROR: there is no such pair (aircraftId, timeId) as ({aircraft_id}, {time_id}) in the sensor database')
        print('exiting...')
        return

    # Join the aircraft_utilization rdd with the sensor_data rdd
    data = aircraft_utilization_rdd.join(
        sensor_data).mapValues(lambda v: (*v[0], v[1]))
    data = data.map(lambda t: t[1])  # Keep only the values
    data = data.toDF(['FH', 'FC', 'DM', 'SensorAVG'])
    # Now, in data we should have the row with the metrics for the pair (aircraft_id, time_id)

    # Prepare the data
    data = data.withColumn("FH", F.col("FH").cast("float"))
    data = data.withColumn("FC", F.col("FC").cast("int"))
    data = data.withColumn("DM", F.col("DM").cast("int"))
    data = data.withColumn("SensorAVG", F.col("SensorAVG").cast("float"))

    # Load the model
    model = PipelineModel.load(model_path)
    # Pass data through the model
    prediction = model.transform(data)
    # Show the prediction
    predicted_label = prediction.collect()[0]['predictedLabel']
    print(
        f'The prediction for aircraftId {aircraft_id} and timeId {time_id} is {predicted_label}')

    return


if __name__ == "__main__":
    HADOOP_HOME = "./resources/hadoop_home"
    JDBC_JAR = "./resources/postgresql-42.2.8.jar"
    PYSPARK_PYTHON = "python3.8"
    PYSPARK_DRIVER_PYTHON = "python3.8"
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

    if os.path.isdir('classification_tree.model'):
        process(sc, spark, 'classification_tree.model')
    else:
        print('ERROR: the model does no exist')
        print('exiting...')
        exit()
