-- Databricks notebook source
-- MAGIC %md 
-- MAGIC
-- MAGIC # Databricks Coding Challenge - SQL
-- MAGIC ### Note: All questions should be done using SQL language
-- MAGIC
-- MAGIC ## Spark SQL and DataFrames 
-- MAGIC
-- MAGIC In this section, you'll read in data to create a DataFrame in Spark.  We'll be reading in a dataset stored in the Databricks File System (DBFS).  Please see this [link](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html#databricks-file-system-dbfs) for more details on how to use DBFS.

-- COMMAND ----------

-- MAGIC %md ##Understanding the data set 
-- MAGIC
-- MAGIC ###Overview:
-- MAGIC The data set used throughout the coding assessment resembles telemetry data that any software as a service (SaaS) company might collect. One record represents the node hours for a single workload running on a transient cluster aggregated at the date and workload type level. This data set may be used to help Databricks understand consumption patterns and user behaviors on our platform. For example, we can inspect this data to understand if a given customer prefers our `automated` or `interactive` features, or understand which AWS instance types are preferred among all of our customers. 
-- MAGIC
-- MAGIC ###Format: 
-- MAGIC  * JSON
-- MAGIC  * Resides on S3
-- MAGIC
-- MAGIC ###Schema:
-- MAGIC * date (String)
-- MAGIC * nodeHours (Double)
-- MAGIC * workloadType (String) (read more [here](https://databricks.com/product/aws-pricing#clusters))
-- MAGIC * metadata (Struct)
-- MAGIC  * clusterMetadata (Struct): Describes the cluster configuration
-- MAGIC  * runtimeMetadata (Struct): Describes the software configuration
-- MAGIC  * workloadMetadata (Struct): Describes the customer. Each shard may have one or many workspaces and each workspace may have zero or many clusters 
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### Part A: SparkSQL and Dataframes 
-- MAGIC
-- MAGIC In this section, you'll read in data to create a dataframe in Spark.  We'll be reading in a dataset stored in the Databricks File System (DBFS).  Please see this link for more details on how to use DBFS:
-- MAGIC https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html#databricks-file-system-dbfs

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC Execute the command below to list the files in a directory that you will be analyzing.  There are several files in this test dataset.

-- COMMAND ----------

-- MAGIC %fs ls /databricks-coding-challenge/workloads

-- COMMAND ----------

-- MAGIC %fs head dbfs:/databricks-coding-challenge/workloads/part-00000-tid-7467717951814126607-30bac750-dd23-4160-a2a6-e57064ff0dc6-1506091-1-c000.json

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### Question 1 (15 points):
-- MAGIC Please create a temporary Spark SQL view called "workloads" from the json files in the directory listed up above

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = (spark.read
-- MAGIC   .format("json")
-- MAGIC   .option("header", "true")
-- MAGIC   .option("inferSchema", "true")
-- MAGIC   .load("/databricks-coding-challenge/workloads/")
-- MAGIC )
-- MAGIC df.createOrReplaceTempView("workloads")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC What is the schema for this table?

-- COMMAND ----------

-- use describe command to get schema of this table
-- alteratively can describe in python using df.printSchema()
desc workloads

-- COMMAND ----------

-- Get an idea of overall size of the table 
-- SELECT COUNT(*) FROM workloads

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### Question 2 (15 points):
-- MAGIC
-- MAGIC Please print out all the unique workspaceId's for this dataset and order them such that workspaceId's are increasing in number.

-- COMMAND ----------

-- unique workspace id's in ascending order
-- Q: workspaceId is string containing int values, do we need to order by their actual value? 
SELECT DISTINCT metadata.workloadMetadata.workspaceId
FROM workloads
ORDER BY abs(workspaceId)  --order by increasing in number

-- COMMAND ----------

-- DBTITLE 1,Sanity checks 
-- sanity check 
-- Q: Why is no of unique workspaceId so less 
-- A: distribution is skewed, most are mapped to one workspaceId
SELECT metadata.workloadMetadata.workspaceId,
       count(*) as no_of_records
FROM workloads
GROUP BY metadata.workloadMetadata.workspaceId
ORDER BY no_of_records DESC

-- COMMAND ----------

--sanity check 2 
SELECT sum(no_of_records) as total_records
FROM (
  SELECT metadata.workloadMetadata.workspaceId,
       count(*) as no_of_records
  FROM workloads
  GROUP BY metadata.workloadMetadata.workspaceId
  ORDER BY no_of_records DESC
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### Question 3 (15 points):
-- MAGIC
-- MAGIC What is the number of unique clusters in this data set?  A cluster is identified by the `metadata.workloadMetadata.clusterId` field.

-- COMMAND ----------

-- no of unique clusters in this data set 
SELECT COUNT(DISTINCT metadata.workloadMetadata.clusterId) AS unique_clusters 
FROM workloads

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Question 4 (15 points): 
-- MAGIC What is the number of workload hours each day for the workspaceID - `-9014487477555684744`?

-- COMMAND ----------

-- no of workload hours each day for workspaceID - -9014487477555684744 
-- Assumption: using nodeHours as a proxy for workload hours 
SELECT date,
       SUM(nodeHours) AS workload_hours
FROM workloads 
WHERE metadata.workloadMetadata.workspaceId = '-9014487477555684744'
GROUP BY date 
ORDER By date

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC Determine how many nodes are spot vs. on demand for a given cluster.

-- COMMAND ----------

-- Get some cluster id which have both
select metadata.workloadMetadata.clusterId,
       collect_set(metadata.clusterMetadata.containerIsSpot) as value_set
from workloads 
group by metadata.workloadMetadata.clusterId
having size(value_set) = 2

-- COMMAND ----------

-- #  change the cluster id here to change the output in cmd24 
CREATE WIDGET TEXT clusterId DEFAULT "-1048771871094449110"

-- COMMAND ----------

-- Determine how many nodes are spot vs. on demand for a given cluster.
-- Sol: group by metadata.clusterMetadata.containerIsSpot for a given cluster id 

SELECT CASE WHEN metadata.clusterMetadata.containerIsSpot THEN 'spot'
            ELSE 'on-demand'
       END AS node_type,
       count(*) as no_of_nodes
FROM workloads WHERE metadata.workloadMetadata.clusterId = $clusterId
GROUP BY metadata.clusterMetadata.containerIsSpot

-- COMMAND ----------

REMOVE WIDGET clusterId

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Question 5 (15 points): 
-- MAGIC
-- MAGIC How many interactive node hours per day are there on the different Spark versions over time.

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Question 6 (25 points):
-- MAGIC #### TPC-H Dataset
-- MAGIC You're provided with a Line Items records from the TPC-H data set. The data is located in `/databricks-datasets/tpch/data-001/lineitem`.
-- MAGIC Find the top two most recently shipped (shipDate) Line Items per Part using the simplest and most efficient approach.
-- MAGIC
-- MAGIC You're free to use any combinations of SparkSQL, PySpark or Scala Spark to answer this challenge.

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ![](https://docs.deistercloud.com/mediaContent/Databases.30/TPCH%20Benchmark.90/media/tpch_schema.png?v=0)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC src ='/databricks-datasets/tpch/data-001/lineitem/lineitem.tbl'
-- MAGIC schema =", ".join(['orderkey int', 'partkey int', 'suppkey int', 'lineNumber int', 'quantity int', 'extendedprice float', 'discount float', 'tax float', 'returnflag string', 'linestatus string', 'shipdate date', 'commitdate date', 'receiptdate date', 'shipinstruct string', 'shipmode string', 'comment string'])
-- MAGIC tpc_h = (spark.read.format("csv") 
-- MAGIC           .schema(schema)
-- MAGIC           .option("header", False)
-- MAGIC           .option("sep", "|")
-- MAGIC           #.option("inferSchema", True)
-- MAGIC           .load(src)
-- MAGIC         )
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.head('/databricks-datasets/tpch/data-001/lineitem/lineitem.tbl')

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(tpc_h)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC tpc_h.createOrReplaceTempView('tpc_h')

-- COMMAND ----------

-- MAGIC %md Find the top two most recently shipped (shipDate) Line Items per Part using the simplest and most efficient approach.

-- COMMAND ----------

-- Q: How much info do we need to return for line item. Is order key enough? 

WITH line_item_ranked AS 
(
SELECT orderkey,
       partkey,
       shipdate,
       -- assuming if there are two shipped on the same date, then only those 2 are returned
       row_number() OVER(PARTITION BY partkey ORDER BY shipDate DESC) AS most_recent_ranked
FROM tpc_h
)
SELECT partkey,
       orderkey,
       shipdate 
FROM line_item_ranked 
WHERE most_recent_ranked <= 2 

-- COMMAND ----------


