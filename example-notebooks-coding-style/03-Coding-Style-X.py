# Databricks notebook source
# MAGIC %md # Databricks Coding Challenge - Coding Style

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1 
# MAGIC #### Part 1: Code analysis and documentation (15 Points)
# MAGIC
# MAGIC In the following cells is code to generate a synthetic data set.  At each point that is marked by commenting blocks ( '#', '"""', '''''), fill in appropriate comments that explain the functionality of each part of the subsequent code in <b><i>standard python code style</i></b>.

# COMMAND ----------

import collections

"""
Named tuples  add the ability to access fields by name instead of position index

Declare named tuples DataStructure, ModuloResult, ConcurrentStructure
"""

DataStructure = collections.namedtuple('DataStructure', 'value1 value2 value3 value4 value5 value6')

ModuloResult = collections.namedtuple('ModuloResult', 'factor remain')

ConcurrentStructure = collections.namedtuple('ConcurrentStructure', 'group value1 value2')

# COMMAND ----------

from pyspark.sql.types import DoubleType, StructType
from pyspark.sql.functions import lit, col
from pyspark.sql import DataFrame
import random
import numpy
from functools import reduce
import math
import string


DISTINCT_NOMINAL = 5
STDDEV_NAME = "_std"
  
class DataGenerator:
  
  def __init__(self, DISTINCT_NOMINAL, STDDEV_NAME): 
    self.DISTINCT_NOMINAL = DISTINCT_NOMINAL
    self.STDDEV_NAME = STDDEV_NAME
  
  def modeFlag(self, mode: str):
    """ 
    Returns mode value for given mode 
    
    Args:
        mode: mode name  
        
    Returns: False/True/None for mode as ascending/ descending/ any other value
    
    """
    
    modeVal = {
      "ascending" : False,
      "descending" : True
    }
    return modeVal.get(mode)
  
  
  def lfold(self, func, nums, exp):
    """
    Returns a list by applying reduce on values to the left of a given number in list
    
    Args:
        func: fucntion to use to reduce 
        nums: An iterable sequence 
        exp: Value placed before the iterable; default value if iterable is empty
        
    Returns: list with reduced values after application of func 
    """
    
    acc = []
    for i in range(len(nums)):
      result = reduce(func, nums[:i+1], exp)
      acc.append(result)
    return acc
  
  
  def generateDoublesData(self, targetCount: int, start: float, step: float, mode: str):
  
    """ 
    Generate sorted doubles list given start, step, targetCount
    
    The method generates a doubles list using start, step, targetCount and then sort 
    the list based on the mode value provided. If mode value is not amongst 
    (ascending/ descending/ random) throws an Exception
    
    Args:
        start: start value of the list 
        step: value to step by when generating the next number 
        targetCount: number of values to be generated 
        mode: order for sorting the list 
        
    Returns: sorted/shuffled doubles list
    """
    
    stoppingPoint = (targetCount * step) + start
    doubleArray = list(numpy.arange(start, stoppingPoint, step))
    try : 
      doubleArray = sorted(doubleArray, reverse=self.modeFlag(mode))
    except:
      if (mode == 'random'):
        random.shuffle(doubleArray)
      else: 
        raise Exception(mode, " is not supported.")
    
    return doubleArray
  
  
  def generateDoublesMod(self, targetCount: int, start: float, step: float, mode: str, exp: float):
    
    """
    Returns DoublesMod of sorted list generated using targetCount,start, step 
    
    The function first generated a sorted list using start, step, targetCount and mode provided 
    and then calculates DoublesMod of the list using the reduce function defined 
    
    Args:
        start: start value of the list 
        step: value to step by when generating the next number 
        targetCount: number of values to be generated 
        mode: order for sorting the list 
        exp: Value placed before the iterable; default value if iterable is empty
    
    Returns: DoublesMod of the generated list
    """

    doubles = self.generateDoublesData(targetCount, start, step, mode)
    res = (lambda x, y: x + ((x + y) / x))
    
    return self.lfold(res, doubles, exp)
    
  
  def generateDoublesMod2(self, targetCount: int, start: float, step: float, mode: str):
    
    """ 
    Returns DoublesMod2 of sorted list generated using targetCount,start, step 
    
    The function first generated a sorted list using start, step, targetCount and mode provided 
    and then calculates DoublesMod2 by first calculating sequenceEval and then calculating a lfold 
    using sequenceEval as value placed before iterable 
    
    Args:
        start: start value of the list 
        step: value to step by when generating the next number 
        targetCount: number of values to be generated 
        mode: order for sorting the list 
    
    Returns: DoublesMod2 of the generated list
    """
    
    doubles = self.generateDoublesData(targetCount, start, step, mode)
     
    func = (lambda x, y: (math.pow((x-y)/math.sqrt(y), 2)))
    sequenceEval = reduce(func, doubles, 0)
    
    res = (lambda x, y: (x + (x / y)) / x)
    return self.lfold(res, doubles, sequenceEval)
  
  
  def generateIntData(self, targetCount: int, start: int, step: int, mode: str):
  
    """
    Generate sorted int list given start, step, targetCount
    
    The method generates an int list using start, step, targetCount and then sort 
    the list based on the mode value provided. If mode value is not amongst 
    (ascending/ descending/ random) throws an Exception
    
    Args:
        start: start value of the list 
        step: value to step by when generating the next number 
        targetCount: number of values to be generated 
        mode: order for sorting the list 
        
    Returns: sorted/shuffled int list
    """
    
    stoppingPoint = (targetCount * step) + start
    intArray = list(range(start, stoppingPoint, step))
    try : 
      intArray = sorted(intArray, reverse=self.modeFlag(mode))
    except:
      if (mode == 'random'):
        random.shuffle(intArray)
      else: 
        raise Exception(mode, " is not supported.")
    
    return intArray

  
  def generateRepeatingIntData(self, targetCount: int, start: int, step: int, mode: str, distinctValues: int): 
    
    """
    Generate sorted int repeating list given start, step, targetCount
    
    The method generates an int list using start, step, targetCount and then sort 
    the list based on the mode value provided, then generates a new array of specified shape and fill_values,
    flatten it and return values till targetCount. If mode value is not amongst 
    (ascending/ descending/ random) throws an Exception
    
    Args:
        start: start value of the list 
        step: value to step by when generating the next number 
        targetCount: number of values to be returned 
        mode: order for sorting the list
        distinctValues: number of values to be generated
        
    Returns: sorted/shuffled int list
    """
     
    subStopPoint = (distinctValues * step) + start - 1
    distinctArray = list(range(start, subStopPoint, step))
    try : 
      sortedArray = sorted(distinctArray, reverse=self.modeFlag(mode))
    except:
      if (mode != 'random'):
        raise Exception(mode, " is not supported.")

    outputArray = numpy.full((int(targetCount / (len(sortedArray) - 1)), len(sortedArray)), 
                                                                 sortedArray).flatten().tolist()[:targetCount]
    if (mode == 'random'):
        random.shuffle(outputArray)
        
    return outputArray
    
    
  def getDoubleCols(self, schema: StructType):
    
    """ 
    Returns a list of col names of double type 
    
    Args:
      schema: schema of a dataframe
      
    Returns: list of col names of double type
    """
    
    return [s.name for s in schema if s.dataType == DoubleType()]
  

  def normalizeDoubleTypes(self, df: DataFrame):

    """ 
    Returns a dataframe  with normalized values of double type cols 
    
    Args:
        df: given DataFrame 
        
    Returns: dataframe  with normalized values of double type cols
    """
    doubleTypes = self.getDoubleCols(df.schema)
    stddevValues = df.select(doubleTypes).summary("stddev").first()
    
    for indx in range(0, len(doubleTypes)):
      df = df.withColumn(doubleTypes[indx]+STDDEV_NAME, col(doubleTypes[indx])/stddevValues[indx+1])
    return df
  
    
  def generateData(self, targetCount: int):
      
      """ 
      Create and return a dataframe with normalized values for double type cols 
      
      The function creates multiple cols using the functions specified, combines them 
      to create a dataframe, normalizes the columns with double type and returns the 
      created dataframe 
      
      Args:
        targetCount: length of the dataframe to be generated 
        
      Returns: dataframe with normalized values for double type cols
      """

      seq1 = self.generateIntData(targetCount, 1, 1, "ascending")
      seq2 = self.generateDoublesData(targetCount, 1.0, 1.0, "descending")
      seq3 = self.generateDoublesMod(targetCount, 1.0, 1.0, "ascending", 2.0)
      seq4 = list(map(lambda x: x * -10, self.generateDoublesMod2(targetCount, 1.0, 1.0, "ascending")))
      seq5 = self.generateRepeatingIntData(targetCount, 0, 5, "ascending", DISTINCT_NOMINAL)
      seq6 = self.generateDoublesMod2(targetCount, 1.0, 1.0, "descending")
      
      seqData: List[DataStructure] = []
        
      for i in range(0, targetCount):
        seqData.append(DataStructure(value1=seq1[i], value2=seq2[i].item(), value3=seq3[i].item(), value4=seq4[i].item(), 
                                      value5=seq5[i], value6=seq6[i].item()))
        
      return self.normalizeDoubleTypes(spark.createDataFrame(seqData))

    
  def generateCoordData(self, targetCount: int):
      
      """ 
      Returns the given dataframe with renamed normalized columns 
      
      The function generated a dataframe using generateData method, renames the 
      normalized columsn and returns the changed dataframe 
      
      Args:
          targetCount: length of the dataframe to be generated
          
      Returns: given dataframe with renamed normalized columns
      """
      
      coordData = self.generateData(targetCount).withColumnRenamed("value2_std", "x1").withColumnRenamed("value3_std", "x2").withColumnRenamed("value4_std", "y1").withColumnRenamed("value6_std", "y2").select("x1", "x2", "y1", "y2")
      return coordData
    
    
  def generateStringData(self, targetCount: int, uniqueValueCount: int):
    
    """ 
    Generates and returns a list of unique keys of length targetCount 
    
    Args:
        targetCount: length of the list to be generated 
        uniqueValueCount: no of unique keys in the list 
        
    Returns: List of length targetCount and unique keys uniqueValueCount 
    """
    orderedColl =  list(string.ascii_lowercase)
    factor = int(targetCount/len(orderedColl))
    remain = targetCount%len(orderedColl)
    uniqueBuffer = list(string.ascii_lowercase)

    if(uniqueValueCount > len(orderedColl)):
      for x in range(0, factor):
        for y in orderedColl:
          uniqueBuffer.append(y + str(x))

    uniqueArray = uniqueBuffer[:uniqueValueCount]

    if (uniqueValueCount > len(orderedColl)):
      uniqueArrayFactor = uniqueArray * factor
      uniqueArrayFactor.extend(uniqueArray[:remain]) 
      utputArray = uniqueArrayFactor
    else:
      outputArray = uniqueArray * int(targetCount / (uniqueValueCount - 1))

    return outputArray[:targetCount]
  
  
  def generateConcurrentData(self, targetCount: int, uniqueKeys: int):
      
      """ 
      Generates and returns a concurrent dataframe 
      
      Generates and returns a concurrent dataframe with group key, standardized doubles data sorted asc, 
      standardized doubles data sortted descending as columns
      
      Args:
          targetCount: length of the dataframe to be generated 
          uniqueKeys: number of unique group keys 
          
      Returns: concurrent dataframe
      """
    
      seq1 = self.generateStringData(targetCount, uniqueKeys)
      seq2 = self.generateDoublesData(targetCount, 1.0, 5.0, "ascending")
      seq3 = self.generateDoublesData(targetCount, 0.0, 0.1, "descending")

      seqData: List[ConcurrentStructure] = []

      for i in range(0, targetCount):
         seqData.append(ConcurrentStructure(group=seq1[i], value1=seq2[i].item(), value2=seq3[i].item()))
      
      return spark.createDataFrame(seqData)

# COMMAND ----------

# MAGIC %md ####Part 2:  Data Skewness (25 Points)
# MAGIC Many data manipulation tasks require the identification and handling of skewed data. This is particularly important for ensuring optimal Spark jobs. In this section, examine the data set that is generated and write a function that will determine the skewness of each column. The only distribution types that are required to be detected are:
# MAGIC - Evenly Distributed
# MAGIC - Left Tailed
# MAGIC - Right Tailed
# MAGIC ######The return type of this function should be a Dictionary of (ColumnName -> Distribution Type)

# COMMAND ----------

dataGenerator = DataGenerator(DISTINCT_NOMINAL, STDDEV_NAME)
data = dataGenerator.generateData(1000)

# COMMAND ----------

columnsToCheck = ["value2_std", "value3_std", "value4_std", "value6_std"]

# COMMAND ----------

from scipy.stats import skew
from typing import List

import pandas
import pyspark 

def determine_skewness(colsToCheck: List[str], data: pyspark.sql.dataframe.DataFrame) -> dict:
  """
  The fucntion returns skewness for each column in the given column list 
  
  Args: 
      colsToCheck: List of columns for which skewness is to be checked 
      data: dataframe containing the columns 
  """
  output_dict = {}
  for col in colsToCheck:
    val_list = data.select(col).rdd.map(lambda x: x).collect()
    
    try:
      skewness = skew(val_list, nan_policy='raise')
    except:
      raise
      
    if skewness == 0:
      output_dict[col] = 'Evenly Distributed'
    elif skewness > 0: 
      output_dict[col] = 'Right Tailed'
    else:
      output_dict[col] = 'Left Tailed'
  
  return output_dict
  

# COMMAND ----------

result_dict = determine_skewness(columnsToCheck, data)
result_dict

# COMMAND ----------

# MAGIC %md ####Part 3: Testing (15 Points)
# MAGIC In order to validate that the function that you have written performs as intended, write a simple test that could be placed in a unit testing framework.
# MAGIC - Demonstrate that the test passes while validating proper classification of at maximum 1 type of distribution
# MAGIC - Demonstate the test failing at classifying correctly, but ensure that the application continues to run (handle the exception and report the failure to stdout)
# MAGIC ######(Hint: Distribution characteristics may change with the number of rows generated based on the data generator's equations)

# COMMAND ----------

def test_skewness():
  # Demonstrate that the test passes while validating proper classification of at maximum 1 type of distribution
  data_5 = dataGenerator.generateData(5)
  cols_to_test = ['value4_std']
  
  expected_outcome = 'Left Tailed'
  actual_outcome = determine_skewness(cols_to_test, data)

  assert actual_outcome['value4_std'] == expected_outcome

test_skewness()

# COMMAND ----------

def test_skewness_failure():
  data_5 = dataGenerator.generateData(5)
  data_5 = data_5.withColumn("test_col", lit(0))
  
  expected_outcome = 'Right Tailed'
  actual_outcome = determine_skewness(['test_col'], data_5)
 
  try:
    assert actual_outcome['test_col'] == expected_outcome
  except:
    print("Assertion Failed: outcome does not meet expected value")
  
test_skewness_failure()

# COMMAND ----------

# MAGIC %md #### Part 4: Efficient Calculations (15 Points)
# MAGIC In this section, create a function that allows for the calculation of euclidean distance between the pairs (x1, y1) and (x2, y2).  Choose the approach that scales best to extremely large data sizes.
# MAGIC - Once complete, determine the distribution type of your derived distance column using the function you developed above in Part 2.
# MAGIC - Show a plot of the distribution to ensure that the distribution type is correct.

# COMMAND ----------

coordData = dataGenerator.generateCoordData(1000)

# COMMAND ----------

display(coordData)

# COMMAND ----------

from pyspark.sql.types import FloatType
from scipy.spatial import distance
import numpy as np

@udf(returnType=FloatType())
def euclidean_distance(x1, y1, x2, y2) -> float:
  """
   The function returns euclidean distance given 4 co-ordinates (x1, y1, x2, y2)
   
   Args:
       x1: x coordiante of point 1
       y1: y coordinate of point 1 
       x2: x coordinate of point 2 
       y2: y coordinate of point 2
  """
  dist = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
  return dist 

# COMMAND ----------

from pyspark.sql import functions as F
euc_coordData = coordData.withColumn("euclidean_distance", euclidean_distance( F.col("x1"),
                                                                                F.col("y1"),
                                                                                F.col("x2"),
                                                                                F.col("y2")
                                                                              ))
display(euc_coordData)

# COMMAND ----------

dist_dict = determine_skewness(["euclidean_distance"], euc_coordData)
dist_dict

# COMMAND ----------

display(euc_coordData.select("euclidean_distance"))

# COMMAND ----------

# MAGIC %md ####Part 5: Complex DataTypes (15 Points)
# MAGIC In this section, create a new column that shows the mid-point coordinates between the (x1, y1) and (x2, y2) values in each row.
# MAGIC - After the new column has been created, write a function that will calculate the distance from each pair (x1, y1) and (x2, y2) to the mid-point value.
# MAGIC - Once the distances have been calculated, run a validation check to ensure that the expected result is achieved.

# COMMAND ----------

from pyspark.sql.functions import struct
from pyspark.sql.types import ArrayType,StructField,StructType,DoubleType

struct_schema_mid = StructType([
    StructField("mid_x", DoubleType()),
    StructField("mid_y", DoubleType())
])

struct_schema_point1 = StructType([
    StructField("x", DoubleType()),
    StructField("y", DoubleType())
])

struct_schema_point2 = StructType([
    StructField("x", DoubleType()),
    StructField("y", DoubleType())
])

euc_coordData = ( euc_coordData.withColumn("mid_point", struct(
                                                                (F.col("x2") + F.col("x1"))/2, 
                                                                (F.col("y1") + F.col("y2"))/2
                                                               ).cast(struct_schema_mid)
                                           )
                                .withColumn("point1", struct(F.col("x1"), F.col("y1")).cast(struct_schema_point1))
                                .withColumn("point2", struct(F.col("x2"), F.col("y2")).cast(struct_schema_point2))
                     )
                 
display(euc_coordData)

# COMMAND ----------

from pyspark.sql.types import MapType,StringType
def get_distance(p1_x,p1_y,p2_x,p2_y) -> float:
  """
  The function returns euclidean distance given 4 co-ordinates (p1_x, p1_y, p2_x, p2_y)
   
   Args:
       p1_x: x coordiante of point 1
       p1_y: y coordinate of point 1 
       p2_x: x coordinate of point 2 
       p2_y: y coordinate of point 2
     
   Returns: Euclidean distance between given points 
  """
  return ((p2_x-p1_x)**2 + (p2_y-p1_y)**2)**(1/2)

@udf(returnType=MapType(StringType(), FloatType()))
def calc_eucl_dist_from_mid(point1, point2, mid_point) -> dict:
  """
   The method calculates and returns distance of point1 and point2 from mid-point 
   
   Args:
       point1: coordinates of point 1 
       point2: coordinates of point 2
       mid_point: coordinates of mid point 
       
   Returns: Distance of point1 and point2 from mid point 
  """
  dist_point1 = get_distance(point1.x, point1.y, mid_point.mid_x, mid_point.mid_y)
  dist_point2 = get_distance(point2.x, point2.y, mid_point.mid_x, mid_point.mid_y)
  return {"dist_point1": dist_point1, "dist_point2": dist_point2}

# COMMAND ----------

euc_coordData = euc_coordData.withColumn("distances", calc_eucl_dist_from_mid(F.col("point1"),
                                                                                        F.col("point2"),
                                                                                        F.col("mid_point")
                                                                                   )
                                    )
display(euc_coordData)

# COMMAND ----------

def validation_check(distances):
  dist1 = [val["dist_point1"] for val in distances.rdd.map(lambda x:x[0]).collect()]
  dist2 = [val["dist_point2"] for val in distances.rdd.map(lambda x:x[0]).collect()]
  assert dist1 == dist2 
  
try:
  validation_check(euc_coordData.select("distances"))
except:
  print("values don't match")
  raise

# COMMAND ----------

# MAGIC %md ####Part 6: Precision (15 Points)
# MAGIC - How many rows of data do not match?
# MAGIC - Why would they / wouldn't they match?

# COMMAND ----------

euc_coordData = euc_coordData.withColumn("recalculatedData", F.col("distances.dist_point1")+F.col("distances.dist_point2"))
df_not_match = euc_coordData.filter(F.col("euclidean_distance") != F.col("recalculatedData"))
print("No of not matching rows: ", df_not_match.count())


# COMMAND ----------

# MAGIC %md
# MAGIC For the data above all the rows match and the precision is 100%. This makes sense as the sum of distances from the midpoint should be equal to the sum of distance between two types. Since we are using double type to store the result we get accurate match as well. Results might differ slightly if we use int data type to store the distance due to rounding off taking place 

# COMMAND ----------

# MAGIC %md ### Optional: Concurrency (25 Bonus Points)
# MAGIC In this section, a data set will be generated for you with a grouping key in the first column.
# MAGIC The desired output of for the final part is a collection in the form of Map[String, Double] that represents the <i><b>mean difference</b></i> between the two numeric fields generated (value1 and value2) <i>within each group</i>.

# COMMAND ----------

conData = dataGenerator.generateConcurrentData(5000000, 12)
display(conData)

# COMMAND ----------

# MAGIC %md ####Part 1: Create a collection for all of the distinct values of the grouping column.  Ensure it is ordered.

# COMMAND ----------

collection = conData.select("group").distinct().orderBy("group").rdd.map(lambda x:x[0]).collect()
collection

# COMMAND ----------

# MAGIC %md ####Part 2: Write a function that will: 
# MAGIC - iterate over each of the unique key values from part 1
# MAGIC - calculate the mean values within each of the groups ("group" column)
# MAGIC - produce the absolute difference between the averages of value 1 and value 2 
# MAGIC - and finally return these values in the form of Map[String, Double] ("group" -> mean difference)

# COMMAND ----------

import numpy as np

def calc_mean_diff(df, unique_vals) -> dict:
  """
    The method returns mean difference of value1 and value2 for each group 
    
    Args:
      df: dataframe object 
      unique_vals: distinct values in group column 
      
    Returns: mean difference of value1 and value2 for each group
  """
  result_dict = {}
  for val in unique_vals:
    mean_val1 = np.mean(df.filter(df.group == val).select("value1").rdd.map(lambda x:x[0]).collect())
    mean_val2 = np.mean(df.filter(df.group == val).select("value2").rdd.map(lambda x:x[0]).collect())
    abs_diff = abs(mean_val2 - mean_val1)
    result_dict[val] = abs_diff
    
  return result_dict

result = calc_mean_diff(conData, collection)
result           

# COMMAND ----------

# MAGIC %md #### Part 3: Write a function that will perform the same calculation concurrently in order to get the execution time as low as possible.

# COMMAND ----------

from pyspark.sql import functions as F

def concurrent_mean_calc(df) -> dict:
  """
    The method returns mean difference of value1 and value2 for each group
    
    Args:
      df: dataframe object 
      
    Returns: mean difference of value1 and value2 for each group
  """
  result_dict = (conData.
                  groupBy("group").
                  agg(F.abs(F.mean("value1") - F.mean("value2")).alias("mean_difference")).
                  select("group", "mean_difference").
                  rdd.
                  map(lambda x:{x.group: x.mean_difference}).
                  collect()
                 )
  
  return result_dict

concurrent_mean_calc(conData)

# COMMAND ----------


