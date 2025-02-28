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
-- MAGIC '''
-- MAGIC #Reading from json directory
-- MAGIC #creating a tempview
-- MAGIC #displaying the dataframe
-- MAGIC '''
-- MAGIC workloads_df = spark.read.json("dbfs:/databricks-coding-challenge/workloads") 
-- MAGIC workloads_df.createOrReplaceTempView('workloads')                         
-- MAGIC spark.sql("select * from workloads").display()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC What is the schema for this table?

-- COMMAND ----------

-- MAGIC %python
-- MAGIC workloads_df.printSchema() #printing the schema

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### Question 2 (15 points):
-- MAGIC
-- MAGIC Please print out all the unique workspaceId's for this dataset and order them such that workspaceId's are increasing in number.

-- COMMAND ----------

-- TODO
-- List all the distinct workspace IDs by increasing order
select distinct metadata.workloadMetadata.workspaceId from workloads order by 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### Question 3 (15 points):
-- MAGIC
-- MAGIC What is the number of unique clusters in this data set?  A cluster is identified by the `metadata.workloadMetadata.clusterId` field.

-- COMMAND ----------

-- TODO
-- Count all the distinct cluster IDs
select count(distinct metadata.workloadMetadata.clusterId) as no_of_unique_clusters from workloads;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Question 4 (15 points): 
-- MAGIC What is the number of workload hours each day for the workspaceID - `-9014487477555684744`?

-- COMMAND ----------

-- TODO
-- List all the distinct workspace IDs by increasing order
select date as day,round(sum(nodeHours),2) as workload_hours from workloads where metadata.workloadMetadata.workspaceId = '-9014487477555684744' group by date order by 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC Determine how many nodes are spot vs. on demand for a given cluster.

-- COMMAND ----------

-- TODO
select metadata.workloadMetadata.clusterId,sum(case when metadata.clusterMetadata.containerIsSpot = 'true' then 1 else 0 end) as spot,sum(case when metadata.clusterMetadata.containerIsSpot = 'false' then 1 else 0 end) as on_demand from workloads group by 1;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Question 5 (15 points): 
-- MAGIC
-- MAGIC How many interactive node hours per day are there on the different Spark versions over time.

-- COMMAND ----------

-- TODO
select date as day,metadata.runtimeMetadata.sparkVersion,round(sum(nodeHours),2) as interactive_node_hours from workloads where workloadType = 'interactive' group by 1,2 order by 1

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

-- TODO
--partition by partkey will take care of 'per part' and rank by shipdate should take most recent two shipments
with lineitems as (
  select partkey,linenumber,shipdate,rank() over (partition by partkey order by shipdate desc) as rnk  from tpc_h
)
select partkey,linenumber,shipdate from lineitems where rnk <= 2;

-- COMMAND ----------


