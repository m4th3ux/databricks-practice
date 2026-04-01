import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from databricks.sdk.runtime import *

spark = SparkSession.builder.getOrCreate()

def trim_columns(df, cols):
    for c in cols:
        df = df.withColumn(c, F.trim(F.col(c)))
    return df

def get_last_partition(spark, table_path: str, partition_col: str):
    """
    Retorna a última partição disponível na tabela Delta.
    
    Ex:
    get_last_partition(spark, "databricks_practice.bronze.bronze_table", "dt_ingestion_partition")
    """
    df = spark.table(table_path)
    row = df.agg(F.max(F.col(partition_col)).alias("last")).first()
    return row["last"]

def get_new_partitions(spark, bronze_table: str, silver_table: str, partition_col: str):
    """
    Retorna um DataFrame contendo SOMENTE as partições novas da Bronze
    que ainda não foram carregadas na Silver.
    """
    
    last_bronze = get_last_partition(spark, bronze_table, partition_col)
    last_silver = get_last_partition(spark, silver_table, partition_col)
    
    # Caso Silver esteja vazia
    if last_silver is None:
        return spark.sql(f"""
            SELECT DISTINCT {partition_col}
            FROM {bronze_table}
        """)
    
    # Caso já exista Silver
    return spark.sql(f"""
        SELECT DISTINCT {partition_col}
        FROM {bronze_table}
        WHERE {partition_col} > '{last_silver}'
    """)


def load_bronze_new_partitions(spark, bronze_table: str, silver_table: str, partition_col: str):
    """
    Retorna um DataFrame da Bronze contendo apenas os dados das partições novas.
    """
    new_parts_df = get_new_partitions(spark, bronze_table, silver_table, partition_col)

    df_bronze = spark.table(bronze_table)

    return (
        df_bronze
        .join(new_parts_df, partition_col, "inner")
    )


def clean_mercado(df_mercado):
    df_mercado = trim_columns(df_mercado, df_mercado.columns)
    
    df_mercado = df_mercado.withColumn(
        "currency",
        F.when((F.col("currency") == "FV7002") & (F.col("country_code").isin("DE", "AT", "ES", "PT")), "EUR")
         .when((F.col("currency") == "FV7002") & (F.col("country_code") == "DK"), "DKK")
         .when((F.col("currency") == "FV7002") & (F.col("country_code") == "SE"), "SEK")
         .when((F.col("currency") == "BGN") & (F.col("country_code") == "BG"), "EUR")
         .otherwise(F.col("currency"))
    )
    return df_mercado
