import pyspark
from pyspark.sql import SparkSession
import boto3
import json
import pandas as pd 

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, LongType, TimestampType

def to_pandas_in_batches(spark_df: DataFrame, batch_size: int = 10000, filter_condition=None) -> pd.DataFrame:
    """
    Transform a PySpark DataFrame to a or Pandas DataFrame in batches to avoid memory errors

    Parameters:
        spark_df: Spark DataFrame
        batch_size int: Number of rows to be transformed at a time (default: 10000)
        filter_condition: condition to be used to filter the input dataframe  (default: None)
    """
    
    # Apply filter condition if provided and cache the DataFrame
    if filter_condition:
        spark_df = spark_df.filter(filter_condition)
    spark_df = spark_df.cache()
    total_rows = spark_df.count()
    
    if total_rows == 0:
        spark_df.unpersist()
        return pd.DataFrame()
    
    num_batches = (total_rows + batch_size - 1) // batch_size

    # Get Spark session from the input DataFrame instead of active session
    spark = spark_df.sparkSession
    
    # Assign unique index using zipWithIndex
    rdd_with_index = spark_df.rdd.zipWithIndex()
    
    # Prepare schema with index column
    indexed_schema = StructType(
        [StructField("index", LongType(), False)] + spark_df.schema.fields
    )
    
    # Create indexed DataFrame
    indexed_rows = rdd_with_index.map(lambda row_idx: (row_idx[1],) + tuple(row_idx[0]))
    indexed_df = spark.createDataFrame(indexed_rows, indexed_schema).cache()  # Fixed here
    
    pandas_dfs = []
    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = start + batch_size
        batch_df = indexed_df.filter((col("index") >= start) & (col("index") < end)).drop("index")
        
        try:
            pandas_df = batch_df.toPandas()
        except TypeError as e:
            print(f"TypeError during conversion: {e}")
            timestamp_cols = [
                field.name for field in batch_df.schema.fields 
                if isinstance(field.dataType, TimestampType)
            ]
            if timestamp_cols:
                batch_df = batch_df.select([
                    col(c).cast("string").alias(c) if c in timestamp_cols else col(c) 
                    for c in batch_df.columns
                ])
            pandas_df = batch_df.toPandas()
        pandas_dfs.append(pandas_df)
        print(f"Batch {batch_num + 1} of {num_batches} finished!")
    
    # Cleanup cached DataFrames
    spark_df.unpersist()
    indexed_df.unpersist()
    
    return pd.concat(pandas_dfs, ignore_index=True)
