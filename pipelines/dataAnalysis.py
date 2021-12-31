from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F
import pickle


def process(sc, spark, source_data):
    data = spark.read.format("csv").option("header", "true").load(source_data)

    data = data.withColumn("FH", F.col("FH").cast("float"))
    data = data.withColumn("FC", F.col("FC").cast("int"))
    data = data.withColumn("DM", F.col("DM").cast("int"))
    data = data.withColumn("SensorAVG", F.col("SensorAVG").cast("float"))

    assembler = VectorAssembler(
        inputCols=['FH', 'FC', 'DM', 'SensorAVG'], outputCol="features")
    data = assembler.transform(data)

    labelIndexer = StringIndexer(
        inputCol="Label", outputCol="indexedLabel", stringOrderType='alphabetDesc').fit(data)

    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    dt = DecisionTreeClassifier(
        labelCol="indexedLabel", featuresCol="features")

    inverter = IndexToString(
        inputCol="prediction", outputCol="predictedLabel", labels=['NoMaintenance', 'Maintenance'])

    pipeline = Pipeline(stages=[labelIndexer, dt, inverter])
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    predictionAndLabels = predictions.select("prediction", "indexedLabel").selectExpr(
        "prediction as prediction", "indexedLabel as label").rdd

    multi_metrics = MulticlassMetrics(predictionAndLabels)
    precision_score = multi_metrics.weightedPrecision
    recall_score = multi_metrics.weightedRecall

    print("The recall is", recall_score)
    print("The precision is", precision_score)

    mat = multi_metrics.confusionMatrix().toArray()
    class1recall = mat[0][0]/sum(mat[0])
    class2recall = mat[1][1]/sum(mat[1])
    good_acc = multi_metrics.accuracy
    print('class 1:', class1recall)
    print('class 2:', class2recall)
    print('good metrics accuracy:', good_acc)

    # itd = inverter.transform(predictions)
    # print(itd)
    # itd.select("Label", "predictedLabel", "features").show(5)
    # print(predictions)

    save_path = "classification_tree.model"
    model.write().overwrite().save(save_path)
    return save_path
