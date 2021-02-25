
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.types import IntegerType
# from pyspark.ml.feature import StringIndexer

df = spark.read.options(header='True').csv("train_vali.csv")

df = df.drop('_c0')
for col in df.columns:
	df = df.withColumn(col, df[col].cast(IntegerType()))

colnames = []
outputcols = []
for i in range(1,10):
	colnames.append("categorical_feature_"+str(i))
	outputcols.append("categoryVec"+str(i))

encoder = OneHotEncoder(inputCols=colnames, outputCols=outputcols)
model = encoder.fit(df)
encoded = model.transform(df)
encoded = encoded.drop(*colnames)

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler( inputCols=encoded.columns, outputCol='features')
transform_encoded = assembler.transform(encoded)



split = transform_encoded.randomSplit([0.8,0.2], 1234)

train = split[0]
vali = split[1]

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'label')
lrModel = lr.fit(train)

trainingSummary = lrModel.summary
accuracy = trainingSummary.accuracy
predictions = lrModel.transform(vali)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
