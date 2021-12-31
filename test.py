# Import the needed libraries
import ast
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark
import numpy as np
from functools import reduce
from datetime import datetime as dt
import datetime

from pyspark.mllib.evaluation import MulticlassMetrics


def process(spark_context: pyspark.SparkContext, spark_session: SparkSession):
    predictionAndLabels = spark_context.parallelize([(0.0, 0.0), (0.0, 1.0), (0.0, 0.0), (
        1.0, 0.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 0.0)])
    metrics = MulticlassMetrics(predictionAndLabels)
    print(metrics.recall(1.0))
