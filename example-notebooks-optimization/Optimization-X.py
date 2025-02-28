# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Coding Challenge - Spark Optimization
# MAGIC
# MAGIC **Dataset:**
# MAGIC * This is synthetic data
# MAGIC * Each year's data is roughly the same with some variation for market growth
# MAGIC * We are looking at retail purchases from the top N retailers

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Optimize Me
# MAGIC
# MAGIC **Optimize the query below:**
# MAGIC * The final **DataFrame** should be assigned to **finalDF**
# MAGIC * The final dataset, on disk, should be partitioned by **zip_code** at an optimal size and written to **finalPath**
# MAGIC * The use of temporary files / datasets is prohibited for this exercise
# MAGIC * Caching can be used during development, but not in the final solution
# MAGIC * The final solution should have only one job and two or fewer stages
# MAGIC * Make it execute as fast as possible.<br/>
# MAGIC
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** There are at least nine different problems with the Scala version and at least ten with the Python version. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question (100 points):  7 required and 2 optional 

# COMMAND ----------

# DBTITLE 1,Setup Env
aws_role_id = "AROAUQVMTFU2DCVUR57M2"
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = f"s3a://e2-interview-user-data/home/{aws_role_id}:{user}"

# COMMAND ----------

# MAGIC %fs ls dbfs:/training/global-sales/solutions/2018-fixed.parquet/

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark import StorageLevel

spark.catalog.clearCache()

trxPath = "dbfs:/training/global-sales/solutions/2018-fixed.parquet/"
citiesPath = "dbfs:/training/global-sales/cities/all.parquet/"
finalPath = "{0}/coding-challenge-2.parquet".format(userhome)

spark.conf.set("spark.sql.shuffle.partitions", 4)

class RestClient:
  def lookupCity (self, city, state):
    try:
      import urllib.request as urllib2
    except ImportError:
      import urllib2
    
    url = "http://api.zippopotam.us/us/{}/{}".format(state, city.replace(" ", "%20"))
    json = urllib2.urlopen(url).read().decode("utf-8")
    posA = json.index("\"post code\": \"")+14
    posB = json.index("\"", posA)
    return json[posA:posB]

def fetch(city, state):
  client = RestClient()
  return client.lookupCity(city, state)

fetchUDF = spark.udf.register("fetch", fetch)

citiesDF = spark.read.parquet(citiesPath).cache()

trxDF = spark.read.parquet(trxPath).cache()

finalDF = (trxDF.join((citiesDF
                       .filter(col("state_abv").isNotNull())
                       .cache()
                       .withColumn("zip_code", fetchUDF(col("city"), col("state_abv")))
                       ).hint("broadcast"), "city_id")
                .write.mode("overwrite")
                .partitionBy("zip_code")
                .parquet(finalPath)
          )

# COMMAND ----------

# MAGIC %md
# MAGIC Optimizations done 
# MAGIC 1. Filter citiesDF before joining and cache(). Only 50 records are left after filtering
# MAGIC 2. Use a broadcast join to make join faster 
# MAGIC 3. Cache trxDF when readig to keep it in memory
# MAGIC 4. Remove repartition as not needed 
# MAGIC 5. Reduce the number of shuffle partitions 
# MAGIC 6. Cache after filtering so that udf is applied on smaller number of rows
# MAGIC 7. Removing unnecessary imports ()
# MAGIC
# MAGIC Not allowed to Implement 
# MAGIC 1. KyroSerializer

# COMMAND ----------


