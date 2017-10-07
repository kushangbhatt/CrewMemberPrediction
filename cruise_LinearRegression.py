# Databricks notebook source
spark

# COMMAND ----------

data = sqlContext.sql("select * from cruise_ship")

# COMMAND ----------

display(data)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data.groupBy('Cruise_line').count().show()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol='Cruise_line',outputCol='Cruise_cat')
indexed = indexer.fit(data).transform(data)
display(indexed)

# COMMAND ----------

indexed.printSchema()

# COMMAND ----------

from pyspark.sql.functions import *
df = indexed.select(indexed['Ship_name'], indexed['Cruise_line'], 
                    indexed['Age'].cast("double").alias('Age'),
                    indexed['Tonnage'].cast("double").alias('Tonnage'),
                    indexed['passengers'].cast("double").alias('passengers'),
                    indexed['length'].cast("double").alias('length'),
                    indexed['cabins'].cast("double").alias('cabins'),
                    indexed['passenger_density'].cast("double").alias('passenger_density'),
                    indexed['crew'].cast("double").alias('crew'),
                    indexed['Cruise_cat'])

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.select('Cruise_cat','crew'))

# COMMAND ----------

display(df.select('Age','crew'))

# COMMAND ----------

display(df.select('Tonnage','crew'))

# COMMAND ----------

display(df.select('crew','passengers'))

# COMMAND ----------

display(df.select('crew','length'))

# COMMAND ----------

display(df.select('crew','cabins'))

# COMMAND ----------

display(df.select('crew','passenger_density'))

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

df.columns


# COMMAND ----------

assembler = VectorAssembler(inputCols=['Age',
 'Tonnage',
 'passengers',
 'length',
 'cabins',
 'passenger_density',
 'Cruise_cat'], outputCol='features')

# COMMAND ----------

(split20, split80) = df.randomSplit((0.20, 0.80), seed=100)
testSet = split20.cache()
trainingSet = split80.cache()

# COMMAND ----------

display(trainingSet.describe())

# COMMAND ----------

display(testSet.describe())

# COMMAND ----------

# ***** LINEAR REGRESSION MODEL ****

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml import Pipeline

# COMMAND ----------

# Let's initialize our linear regression learner
lr = LinearRegression()

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# Now we set the parameters for the method
lr.setPredictionCol("Predicted_Crew")\
  .setLabelCol("crew")\
  .setMaxIter(100)

# COMMAND ----------

# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()
lrPipeline.setStages([assembler, lr])

# COMMAND ----------

# Let's first train on the entire dataset to see what we get
lrModel = lrPipeline.fit(trainingSet)# Now we set the parameters for the method

# COMMAND ----------

def to_equation(model):
  # The intercept is as follows:
  intercept = lrModel.stages[1].intercept

  # The coefficents (i.e. weights) are as follows:
  weights = lrModel.stages[1].coefficients.toArray()

  featuresNoLabel = [col for col in df.columns if col not in ["Ship_name",
 "Cruise_line","crew"]]
  coefficents = sc.parallelize(weights).zip(sc.parallelize(featuresNoLabel))
  equation = "y = {intercept}".format(intercept=intercept)
  variables = []

  # Now let's sort the coeffecients from the most to the least and append them to the equation.
  for x in coefficents.sortByKey().collect():
      weight = x[0]
      name = x[1]
      symbol = "+" if (x[0] > 0) else "-"
      equation += (" {} ({} * {})".format(symbol, weight, name))

  # Finally here is our model expressed an equation
  return equation

# COMMAND ----------

print("Linear Regression Equation: " + to_equation(lrModel))

# COMMAND ----------

predictionsAndLabels = lrModel.transform(testSet)
display(predictionsAndLabels)

# COMMAND ----------

# Now let's compute some evaluation metrics against our test dataset
from pyspark.mllib.evaluation import RegressionMetrics
metrics = RegressionMetrics(predictionsAndLabels.select("Predicted_Crew", "crew").rdd.map(lambda r: (float(r[0]), float(r[1]))))

# COMMAND ----------

rmse = metrics.rootMeanSquaredError
explainedVariance = metrics.explainedVariance
r2 = metrics.r2

print("Root Mean Squared Error: {}".format(rmse))
print("Explained Variance: {}".format(explainedVariance))
print("R2: {}".format(r2))

# COMMAND ----------

# First we calculate the residual error and divide it by the RMSE
predictionsAndLabels.selectExpr("crew", "Predicted_Crew", "crew - Predicted_Crew Residual_Error", "(crew - Predicted_Crew) / {} Within_RSME".format(rmse)).registerTempTable("Ship_Crew_RMSE_Evaluation")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from Ship_Crew_RMSE_Evaluation

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now we can display the RMSE as a Histogram. Clearly this shows that the RMSE is centered around 0 with the vast majority of the error within 2 RMSEs.
# MAGIC SELECT Within_RSME  from Ship_Crew_RMSE_Evaluation

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT case when Within_RSME <= 1.0 and Within_RSME >= -1.0 then 1  when  Within_RSME <= 2.0 and Within_RSME >= -2.0 then 2 else 3 end RSME_Multiple, COUNT(*) count  from Ship_Crew_RMSE_Evaluation
# MAGIC group by case when Within_RSME <= 1.0 and Within_RSME >= -1.0 then 1  when  Within_RSME <= 2.0 and Within_RSME >= -2.0 then 2 else 3 end

# COMMAND ----------


