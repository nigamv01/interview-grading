# Databricks notebook source
# MAGIC %md # Databricks Coding Challenge - Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use SF Airbnb rental data to predict nightly Airbnb rental prices in San Francisco.

# COMMAND ----------

# MAGIC %md ###Q1: EDA (20 Points)
# MAGIC <br />
# MAGIC Create a Spark Dataframe from `/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/`.
# MAGIC
# MAGIC Visualize and explore the data. Note anything you find interesting. This dataset is slightly cleansed form of the [Inside Airbnb](http://insideairbnb.com/get-the-data.html) dataset for San Francisco.

# COMMAND ----------

# Installing mlflow for Q2- restarting python will reset the python environment, run this cell first!
dbutils.library.installPyPI("mlflow", "1.12")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load and display dataset
file_path = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"

# Load data set 
airbnb_df = spark.read.parquet(file_path)

#Convert to pandas dataframe for exploratory and other analysis 
airbnb_df_pandas = airbnb_df.toPandas()

# Display the data set 
airbnb_df_pandas.describe()

# COMMAND ----------

# DBTITLE 1,Break the dataset into features and label
# Separate out the label and features 
label = airbnb_df.select("price")
features = airbnb_df.drop("price")

# COMMAND ----------

# DBTITLE 1,Distribution of numerical features
# Let's see the distribution of all numerical variables in the dataset 
features.toPandas().hist(bins=30, figsize=(15, 10))


# COMMAND ----------

# MAGIC %md
# MAGIC From the dataset we can see that most of features are numerical but categorical in nature thereby making case for using tree based models. But let's explore further. 

# COMMAND ----------

# DBTITLE 1,Linear Regression Check
# To check if linear regression can be applied on this dataset we need to do a normality test 
from pyspark.sql.types import DoubleType
import statsmodels.formula.api as smf

# Get a list of double calls 
dbl_cols = [f.name for f in airbnb_df.schema.fields if isinstance(f.dataType, DoubleType)]
dbl_cols_str = '+'.join(dbl_cols)
smf_ols_str = 'price ~ '+dbl_cols_str

# Fit an ols model 
model = smf.ols(smf_ols_str, data = airbnb_df_pandas).fit()

# COMMAND ----------

from statsmodels.stats.diagnostic import lilliefors
import scipy.stats as scs

# Get the model residual 
model_resid = model.resid

# Normality testing code borrowed from (https://colab.research.google.com/gist/gurezende/113ba28efc601070b78f5ac4426736a6/linear-regression-tested.ipynb#scrollTo=kEw-tdjPjrYC)

# For training a linear regression model on the dataset errors should follow a normal distribution. Below we can do a normality test on the errors to identify 
# if we can apply linear regression on the dataset as it is 

# Kolmogorov-Smirnov test
_, p = lilliefors(model_resid, dist='norm')
print("Kolmogorov-Smirnov test: ")
print('Not normal | p-value:' if p < 0.05 else 'Normal | p-value:', p)
print('-------------------------------')

 # Anderson
stat, p3, _ = scs.anderson(model_resid, dist='norm')
print('Anderson:')
print('Not normal | stat:' if stat > p3[2] else 'Normal | stat:', stat, ':: p-value:', p3[2])

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from the test above that errors of a simple OLS model are not normally distributed which breaks a necessary assumption of Linear Regression. Hence we cannot use Linear Regression model if we want to use these features  as it is. There are ways to transform the data for the linear model to be applicable but given the distributions of various features and limited time for this project, I will be moving ahead with Tree Based Models. 

# COMMAND ----------

# DBTITLE 1,Correlation
# Let's explore the correlation amongst different features, and their correlation with label

import numpy as np 

# Code taken from (https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on)

# Create correlation matrix
corr_matrix = airbnb_df_pandas.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95 that can be dropped
highly_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
print("Highly correlated columns: ", highly_corr)

# COMMAND ----------

upper['price'].sort_values(ascending=False)

# features correlated with price in descending order 

# COMMAND ----------

# On exploring visually the correlation matrix we can see that columns in to_drop are highly correlated with review_scores_rating_na
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(12,10))
# cor = X_train.corr()
sns.heatmap(upper, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We see that there are a number of highly correlated features. We can do transformations on the data to make it normal so that it can be used by linear regression models. However, seeing the distribution for features and the limited time for this exercise I  plan to use tree based models. Tree based models are not affected by multicolinearity we don't need to remove these features at this stage. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Q2: Model Develepment and Tracking (80 Points)
# MAGIC
# MAGIC
# MAGIC * Split into 80/20 train-test split using SparkML APIs. (5 points)
# MAGIC * Build a model using SparkML to predict `price` given the other input features (or subset of them). (30 points)
# MAGIC * Mention why you chose this model, how it works, and other models that you considered. (30 points)
# MAGIC * Compute the loss metric on the test dataset and explain your choice of loss metric. (5 points)
# MAGIC * Log your model/hyperparameters/metrics to MLflow. (10 points) <br/>

# COMMAND ----------

# DBTITLE 1,Split into 80/20train-test
trainDF, testDF = airbnb_df.randomSplit([0.8, 0.2], seed=12345)

# COMMAND ----------

# MAGIC %md
# MAGIC Before we go into model building let's set up a baseline model. A simplest baseline model for regression can be when every record is predicted as the average of the price 

# COMMAND ----------

# DBTITLE 1,RMSE for simple base model

avg_price = airbnb_df.groupBy().avg('price').rdd.map(lambda x:x[0]).collect()
pred_price = avg_price * airbnb_df.count()
actual_price = airbnb_df.select("price").rdd.map(lambda x: x[0]).collect()
base_rmse = sum((np.subtract(pred_price,actual_price))**2)**(1/2)
base_rmse

# COMMAND ----------

# Since our dataset has categorical features, we need to convert them into numerical values to be able to use them in our training 
# The technique we will use here is one hot encoding. In this technique a unique column is created for every distinct value of a categorical 
# feature , where rows which have that partiuclar value are marked as 1 and remaining are marked as zero. One-hot encoding is also particularly useful
# in tree based models which we plan to use. More details about one hot encoding can be found here (https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categoricalColumns = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputColumns = [x + "Index" for x in categoricalColumns]
oheOutputColumns = [x + "OHE" for x in categoricalColumns]

# StringIndexer transformation details can be found here (https://spark.apache.org/docs/latest/ml-features.html#stringindexer)
stringIndexer = StringIndexer(inputCols=categoricalColumns, outputCols=indexOutputColumns, handleInvalid="skip")

# One hot encoder transformationd etails can be found here (https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)
oheEncoder = OneHotEncoder(inputCols=indexOutputColumns, outputCols=oheOutputColumns)


# COMMAND ----------

# Now we need to get all our numeric and encoded columns into a Vector Assembler, since Spark ML algorithms need input in that form 
from pyspark.ml.feature import VectorAssembler

# Get one-hot encoded and numeric column names
oheColumns = [c + "OHE" for c in categoricalColumns]

numericColumns = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price"))]

assemblerInputs = oheColumns + numericColumns

# Create the VectorAssembler transformer
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# DBTITLE 1,Loss Metric
# MAGIC %md
# MAGIC For this regression problem we choose Root Mean Square Error (RMSE) as loss metric. Since it is a regression problem we essentially want to find how off the prediction was from the actual value. A good measure for such a case is calculating average euclidean distance between the predicted and the actual values. This in essence is done through RMSE. The aim is to minimize RMSE as much as possible.  A detailed description can be found at (https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/)

# COMMAND ----------

# Build a machine learning algorithm 

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor,RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

import math 

rmse_dict = {}
model_dict = {}

# Define a dict with all the models we want to try 
algo_dict = {
              'decision_tree': DecisionTreeRegressor(featuresCol="features", labelCol="price"),
              'random_forest': RandomForestRegressor(featuresCol="features", labelCol="price"),
              'gradient_boosting': GBTRegressor(featuresCol="features", labelCol="price", maxIter=20)
             }

# Define the evaluator for the problem 
reg_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")

def run_models(trainDF, testDF, algo_dict):
  """
    The method calculates best model with its rmse for a given train and test data set 
    
    Args:
      trainDF: training data set 
      testDF:  test data set
      algo_dict: dictionary of algorithms to be tried 
      
    Returns: Trained model for each algorithm with its RMSE on test set
    
  """
  for algo, algo_obj in algo_dict.items():
      
      pipeline = Pipeline(stages=[stringIndexer, oheEncoder, assembler, algo_obj])

      # Train model.  This also runs the indexer.
      model = pipeline.fit(trainDF)

      predictions = model.transform(testDF)

      rmse = reg_evaluator.evaluate(predictions)
        
      print("Root Mean Squared Error (RMSE) on test data for algo", algo," = ", rmse, "and base_rmse = ", base_rmse)
      if rmse < base_rmse:
        model_dict[algo] = model
        rmse_dict[algo] = rmse
    
  return{'model_dict': model_dict, 'rmse_dict': rmse_dict}

# COMMAND ----------

result = run_models(trainDF, testDF, algo_dict)
result

# COMMAND ----------

# model summary 
print(result['model_dict']['random_forest'].stages[3])

# COMMAND ----------

# DBTITLE 1,Feature Importance
from collections import OrderedDict
feature_imp_values = result['model_dict']['random_forest'].stages[3].featureImportances.toArray()
col_names = assemblerInputs
feature_imp_dict = {x:y for x,y in zip(col_names, feature_imp_values)}
feature_imp_dict = OrderedDict(sorted(feature_imp_dict.items(), key=lambda item: item[1], reverse=True))
feature_imp_dict

# COMMAND ----------

# DBTITLE 1,Model Selection
# MAGIC %md
# MAGIC
# MAGIC ##### Was not able to use crossvalidator, paramgridbuilder, TrainTestValidator or MLFlow due to an error. Details of error are in the cells afterwards 
# MAGIC
# MAGIC #### Ideal Approach
# MAGIC ##### The ideal approach would be to create a Cross Validator along with ParamGrid for each of these algorithms to find the best tuned model for each of the algorithms and then use the testDF to see which algorithm is performing best. In real life scenarios one might also want to test out the trained models over a period of time to catch variation/drift in model predictions. Given the limited amount of data here we are using only the test data set for these calculations 
# MAGIC
# MAGIC Based on rmse on test set, Random Forest regressor is selected. The model is then used to make predictions on test dataset and calculate rmse . 
# MAGIC
# MAGIC How random forest works in regression:
# MAGIC
# MAGIC 1. At a very high level a random forest is an ensemble model using bagging methodology. 
# MAGIC 2. Different bags of data are collected from the training data set by sampling both rows and columns. No of columns in a bag can be varied. A good way to start from can be sqrt(N), where N is the number of columns
# MAGIC 3. A decision tree is then trained on this bag of data. No of bags created is equal to the number of trees one wants to create. 
# MAGIC 4. Split in decision tree is done by using gini index. 
# MAGIC     `How Gini Index is calculated 
# MAGIC 	 Step 1: For a given condition of a node, split the training samples into different branches. 
# MAGIC      
# MAGIC      Step 2: Calculate gini index for each branch. Gini index basically calculates impurity in results of each branch. So in a branch if all samples have the same label it is considered as pure and the gini index will be zero. If there is a 50-50 split of samples in a branch gini index will be 1. Gini index is calculated by 
# MAGIC
# MAGIC                                   Gini = 1 - sigma(1 to c) pi^2
# MAGIC                                           where c-> no of samples in a node after split
# MAGIC       
# MAGIC       Step 3: To calculate the gini index of the node a weighted average of gini index of each branch is calculated where weight of each branch is -> no of samples in branch/ total no of samples
# MAGIC
# MAGIC 5. The final prediction is done by combining output from each of these trees. 
# MAGIC 6. In the case of Regressor, an average of all the predictions are taken. 
# MAGIC 7. Since multiple trees are created on different bags of data it solves the problem of overfitting as well.  
# MAGIC
# MAGIC Other models which were tried are DecisionTreeRegressor and GradientBoostingRegressor

# COMMAND ----------

dt = DecisionTreeRegressor(featuresCol="features", labelCol="price")
pipeline = Pipeline(stages=[stringIndexer, oheEncoder, assembler, dt])
crossval = CrossValidator(estimator=pipeline,
                          evaluator=reg_evaluator,
                          numFolds=2)
crossval.fit(trainDF)

# COMMAND ----------

# DBTITLE 1,Sample modification with MLFlow 
import mlflow
from mlflow import spark
mlflow.pyspark.ml.autolog() # (Remove here when you run on old version of ML runtime. See above.)
with mlflow.start_run():
  model = pipeline.fit(df)
  predictions = model.transform(testDF)
  mlflow.log_metric("rmse", reg_evaluator.evaluate(predictions)); 
  mlflow.spark.log_model(model, "model-file")   

# COMMAND ----------

# DBTITLE 1,Libraries could not use due to Error 
# MAGIC %md
# MAGIC
# MAGIC ### Error
# MAGIC Due to repeated occurence of error 
# MAGIC
# MAGIC TypeError: Descriptors cannot not be created directly.
# MAGIC If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
# MAGIC If you cannot immediately regenerate your protos, some other possible workarounds are:
# MAGIC  1. Downgrade the protobuf package to 3.20.x or lower.
# MAGIC  2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
# MAGIC  
# MAGIC More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates 
# MAGIC
# MAGIC I am not able to use CrossValidation, ParamGrid, TrainTestValidator and MLFlow libraries in this solution. 
# MAGIC
# MAGIC A small sample code for how mlflow could have been used is also given in the cell above.
# MAGIC
# MAGIC ### Ideal Approach 
# MAGIC The ideal approach would be to create a Cross Validator along with ParamGrid for each of these algorithms to find the best tuned model model for each of them and then use the testDF to see which algorithm is performing best. 
# MAGIC
# MAGIC ### Conclusion
# MAGIC Based on the results obtained from our current version RandomForest seem to be performing best 
# MAGIC

# COMMAND ----------


