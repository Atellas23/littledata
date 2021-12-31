# Import the needed libraries
import ast
from pyspark.sql import SparkSession
import pyspark
from functools import reduce
from datetime import datetime as dt
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, IndexToString


def process(spark_context: pyspark.SparkContext, spark_session: SparkSession, model_path: str):
    # Read the user credentials to access the database in PostgresFIB
    aircraft_id = 'XY-LOL'
    time_id = '010312'
    with open('.uname_and_pwd', 'r') as file:
        credentials = ast.literal_eval(file.read())
        username = credentials['username']
        password = credentials['password']

    # Define a function to parse out the date from a string
    def parse_date(timestampstr: str, format: str = '%Y-%m-%d') -> str:
        return str(dt.strptime(timestampstr, format).date())

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
                                .filter(lambda t: t[0] == aircraft_id and parse_date(str(t[1])) == parse_date(time_id, '%d%m%y'))
                                .map(lambda t: ((str(t[1]), t[0]), t[2::]))
                                .mapValues(lambda v: (float(v[0]), int(v[1]), int(v[2])))
                                )

    # Read the files in the resources/trainingData directory and
    # obtain the mean of the sensor data per aircraft per day
    csv_files = (spark_context.wholeTextFiles('resources/trainingData/*')
                 .map(lambda t: (t[0][-30:-4], t[1]))
                 # aquesta línia és nova d'ara
                 .filter(lambda t: t[0][:6] == time_id and t[0][-6:] == aircraft_id)
                 .map(lambda t: ((parse_date(t[0][:6], '%d%m%y'), t[0][-6:]), t[1]))
                 .mapValues(lambda v: v.split('\n')[1:-2:])
                 .mapValues(lambda v: [(float(r), 1) for r in list(map(lambda element: element.split(';')[2], v))])
                 .mapValues(lambda v: tuple(reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]), v)))
                 .mapValues(lambda v: (v[0]/v[1], v[1]))
                 .reduceByKey(lambda a, b: ((a[0]*a[1]+b[0]*b[1])/(a[1]+b[1]), a[1]+b[1]))
                 .mapValues(lambda v: v[0])
                 )

    # Join the aircraft_utilization rdd with the sensor data rdd
    data = aircraft_utilization_rdd.join(
        csv_files).mapValues(lambda v: (*v[0], v[1]))
    data = data.map(lambda t: t[1])
    print(type(data))
    print(data.collect())
    data = data.toDF(['FH', 'FC', 'DM', 'SensorAVG'])

    # Now, in data we should have the row with the metrics for the pair (aircraft_id, time_id)
    # print(data.collect())

    # Prepare the data
    data = data.withColumn("FH", F.col("FH").cast("float"))
    data = data.withColumn("FC", F.col("FC").cast("int"))
    data = data.withColumn("DM", F.col("DM").cast("int"))
    data = data.withColumn("SensorAVG", F.col("SensorAVG").cast("float"))

    assembler = VectorAssembler(
        inputCols=['FH', 'FC', 'DM', 'SensorAVG'], outputCol="features")
    data = assembler.transform(data)

    # Load the model
    model = PipelineModel.load(model_path)
    prediction = model.transform(data)
    prediction.show(5)

    return None
