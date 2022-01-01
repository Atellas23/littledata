# Import the needed libraries
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession


def process(spark: SparkSession, source_path: str, model_path: str, log_path: str = 'data_analysis_metrics.txt'):
    # First we read the processed data as a spark dataframe
    data = spark.read.format("csv").option("header", "true").load(source_path)

    # In order to be able to use VectorAssembler, the features must be numeric
    # So we change the type of each column to the corresponding one
    data = data.withColumn("FH", F.col("FH").cast("float"))
    data = data.withColumn("FC", F.col("FC").cast("int"))
    data = data.withColumn("DM", F.col("DM").cast("int"))
    data = data.withColumn("SensorAVG", F.col("SensorAVG").cast("float"))

    # We use VectorAssembler to create a column "features" containing all the metrics
    # This is done because the decision tree takes only one column as features
    assembler = VectorAssembler(
        inputCols=['FH', 'FC', 'DM', 'SensorAVG'], outputCol="features")

    # We use StringIndexer to change from string labels (Maintenance/NoMaintenance)
    # to binary labels (0 or 1). We assign labels in descending lexicographic order
    labelIndexer = StringIndexer(
        inputCol="Label", outputCol="indexedLabel", stringOrderType='alphabetDesc').fit(data)
    data = labelIndexer.transform(data)

    # We split data into train and test. We use a seed for reproducibility
    seed = 123456
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed)

    # We declare the decision tree model with the target and feature columns that it will use
    dt = DecisionTreeClassifier(
        labelCol="indexedLabel", featuresCol="features")

    # We use IndexToString to change from binary label to string label (Maintenance/NoMaintenance)
    # This will be added to the model pipeline so that the model outputs the predicted semantic
    # label (Maintenance/NoMaintenance) instead of the predicted numeric class (1 or 0)
    inverter = IndexToString(
        inputCol="prediction", outputCol="predictedLabel", labels=['NoMaintenance', 'Maintenance'])

    # This is the main pipeline for the model
    pipeline = Pipeline(stages=[assembler, dt, inverter])

    # Now we train the classifier with the training split
    model = pipeline.fit(trainingData)

    # We get the predictions for the testing set
    predictions = model.transform(testData)

    # In order to get the accuracy, and other metrics, we convert the predictions to an rdd.
    # We leverage this transformation to select only the columns we need and rename them
    predictionAndLabels = predictions.select("prediction", "indexedLabel").selectExpr(
        "prediction as prediction", "indexedLabel as label").rdd

    # We use the MulticlassMetrics class to get all the metrics we need
    multi_metrics = MulticlassMetrics(predictionAndLabels)

    # We get the accuracy of our model when applied to the test dataset
    acc = multi_metrics.accuracy
    # print('Accuracy:', acc)

    # Now we use the confusion matrix to get the recall and precision for each class
    confusion_matrix = np.array(multi_metrics.confusionMatrix().toArray())

    # We calculate the metrics
    class0recall = confusion_matrix[0][0]/sum(confusion_matrix[0])
    class1recall = confusion_matrix[1][1]/sum(confusion_matrix[1])
    class0precision = confusion_matrix[0][0]/sum(confusion_matrix[:, 0])
    class1precision = confusion_matrix[1][1]/sum(confusion_matrix[:, 1])

    # And we output them, both in terminal and in a file
    with open(log_path, 'w') as file:
        output = f'''Metrics:
  - Accuracy = {acc}
  - Recall:
    - Class 0 (NoMaintenance) = {class0recall}
    - Class 1 (Maintenance) = {class1recall}
  - Precision:
    - Class 0 (NoMaintenance) = {class0precision}
    - Class 1 (Maintenance) = {class1precision}'''
        print(output)
        print(output, file=file)

    # Finally, we save the model
    model.write().overwrite().save(model_path)
