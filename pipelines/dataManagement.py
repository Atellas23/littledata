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
                                  .select('aircraftregistration', 'starttime', 'subsystem', 'kind')
                                  .rdd
                                  .filter(lambda t: t[-2] == '3453')
                                  .map(lambda t: ((parse_date(str(t[1])[:19], '%Y-%m-%d %H:%M:%S'), str(t[0])), t[-1]))
                                  .filter(lambda t: t[-1] in ('Delay', 'Safety', 'AircraftOnGround'))
                                  .map(lambda t: t[:-1])
                                  )

    # Read the files in the resources/trainingData directory and
    # obtain the mean of the sensor data per aircraft per day
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

    operation_interruption_rdd = operation_interruption_rdd.flatMap(lambda t: [((str(dt.strptime(
        t[0][0], '%Y-%m-%d').date()-datetime.timedelta(days=d)), t[0][1]), True) for d in range(7)])

    data = data.leftOuterJoin(operation_interruption_rdd).mapValues(
        lambda v: (*v[0], v[1])).mapValues(lambda v: (*v[:-1], 'UnscheduledMaintenance' if v[-1] else 'ScheduledMaintenance'))

    # Return the matrix (save it?)
    with open('result.csv', 'w') as f:
        data = data.values().collect()
        result = 'FH,FC,DM,SensorAVG,Label\n'
        for v in data:
            aux = map(str, v)
            result += ','.join(aux)+'\n'
        print(result, file=f)

    return 'result.csv'
