"""
PySpark feature engineering pipeline.

Reads the ``silver.transactions_labeled`` Iceberg table, computes ML features,
and writes the results to ``gold.ml_features``.

Features produced
-----------------
* transaction_amount         – original numeric value
* transaction_amount_scaled  – standard-scaled version
* hour_of_day                – 0-23
* day_of_week                – 0 (Mon) – 6 (Sun)
* merchant_category_encoded  – ordinal integer via StringIndexer
* device_type_encoded        – ordinal integer via StringIndexer
* txn_count_1h               – rolling transaction count per customer (1-hour window)
* total_amount_1h            – rolling spend sum per customer (1-hour window)
* txn_count_24h              – rolling transaction count per customer (24-hour window)
* total_amount_24h           – rolling spend sum per customer (24-hour window)
* label_encoded              – 0 = non_fraud, 1 = fraud

Usage
-----
    spark-submit --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0 \\
        features/spark_pipeline.py \\
        --catalog-uri http://localhost:8181 \\
        --s3-endpoint http://localhost:9000
"""

import argparse
import logging
import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Spark session ─────────────────────────────────────────────────────────────

def build_spark(catalog_uri: str, s3_endpoint: str, warehouse: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName("fraud-feature-engineering")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.fraud", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.fraud.type", "rest")
        .config("spark.sql.catalog.fraud.uri", catalog_uri)
        .config("spark.sql.catalog.fraud.warehouse", warehouse)
        .config("spark.sql.catalog.fraud.s3.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(spark: SparkSession) -> None:
    df = spark.table("fraud.silver.transactions_labeled")

    # 1. Temporal features
    df = df.withColumn("event_ts", F.to_timestamp(F.col("timestamp").cast("long")))
    df = (
        df
        .withColumn("hour_of_day", F.hour("event_ts"))
        .withColumn("day_of_week", F.dayofweek("event_ts") - 1)  # 0=Mon
    )

    # 2. Rolling windows (per customer_id ordered by timestamp)
    ts_col = F.col("timestamp").cast("long")

    ONE_HOUR = 3600
    TWENTY_FOUR_HOURS = 86400

    w_1h = (
        Window.partitionBy("customer_id")
        .orderBy(ts_col)
        .rangeBetween(-ONE_HOUR, 0)
    )
    w_24h = (
        Window.partitionBy("customer_id")
        .orderBy(ts_col)
        .rangeBetween(-TWENTY_FOUR_HOURS, 0)
    )

    df = (
        df
        .withColumn("txn_count_1h", F.count("transaction_amount").over(w_1h))
        .withColumn("total_amount_1h", F.sum("transaction_amount").over(w_1h))
        .withColumn("txn_count_24h", F.count("transaction_amount").over(w_24h))
        .withColumn("total_amount_24h", F.sum("transaction_amount").over(w_24h))
    )

    # 3. Ordinal encoding via StringIndexer
    indexers = [
        StringIndexer(inputCol="merchant_category", outputCol="merchant_category_encoded",
                      handleInvalid="keep"),
        StringIndexer(inputCol="device_type", outputCol="device_type_encoded",
                      handleInvalid="keep"),
        StringIndexer(inputCol="label", outputCol="label_encoded",
                      handleInvalid="keep", stringOrderType="alphabetAsc"),
    ]

    # 4. Numeric scaling
    assembler = VectorAssembler(inputCols=["transaction_amount"], outputCol="amount_vec")
    scaler = StandardScaler(inputCol="amount_vec", outputCol="amount_scaled_vec",
                            withMean=True, withStd=True)

    pipeline = Pipeline(stages=indexers + [assembler, scaler])
    model = pipeline.fit(df)
    df = model.transform(df)

    # Extract scaled scalar from vector
    df = df.withColumn("transaction_amount_scaled", df["amount_scaled_vec"].getItem(0))

    # 5. Final feature selection
    feature_cols = [
        "customer_id",
        "timestamp",
        "transaction_amount",
        "transaction_amount_scaled",
        "hour_of_day",
        "day_of_week",
        "merchant_category",
        "merchant_category_encoded",
        "device_type",
        "device_type_encoded",
        "txn_count_1h",
        "total_amount_1h",
        "txn_count_24h",
        "total_amount_24h",
        "label",
        "label_encoded",
    ]
    df_features = df.select(feature_cols)

    # 6. Add date partition column
    df_features = df_features.withColumn(
        "date", F.to_date(F.to_timestamp(F.col("timestamp").cast("long")))
    )

    # 7. Write to gold Iceberg table
    (
        df_features.writeTo("fraud.gold.ml_features")
        .partitionedBy("date")
        .createOrReplace()
    )

    count = spark.table("fraud.gold.ml_features").count()
    logger.info("gold.ml_features written successfully. Row count: %d", count)


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="PySpark feature engineering pipeline")
    parser.add_argument("--catalog-uri", default=os.getenv("ICEBERG_CATALOG_URI", "http://localhost:8181"))
    parser.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", "http://localhost:9000"))
    parser.add_argument("--warehouse", default=os.getenv("ICEBERG_WAREHOUSE", "s3a://warehouse/"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    spark = build_spark(args.catalog_uri, args.s3_endpoint, args.warehouse)
    try:
        compute_features(spark)
    finally:
        spark.stop()
