
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.ml.feature import StringIndexer

df1 = spark.read.options(header='True').csv("train.csv")
df2 = spark.read.options(header='True').csv("validation.csv")

def data_transform(df):
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
	return transform_encoded

train = data_transform(df1)
vali = data_transform(df2)

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'label')
lrModel = lr.fit(train)

trainingSummary = lrModel.summary
accuracy = trainingSummary.accuracy
predictions = lrModel.transform(vali)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
