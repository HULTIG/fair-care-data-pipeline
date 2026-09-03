from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit, input_file_name
import hashlib

class DataIngestion:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def ingest(self, source_path: str, output_path: str, dataset_name: str, source_system: str = "manual_upload", has_header: bool = True, column_names: list = None, delimiter: str = ",", drop_columns: list = None):
        """
        Ingests a CSV file into a Bronze Delta table.
        """
        print(f"Ingesting {dataset_name} from {source_path} to {output_path}...")
        
        # Read CSV
        # Using inferSchema for now, but in production we should enforce schema
        header_option = "true" if has_header else "false"
        df = self.spark.read.format("csv") \
            .option("header", header_option) \
            .option("delimiter", delimiter) \
            .option("inferSchema", "true") \
            .load(source_path)
            
        if not has_header and column_names:
            # Check if column count matches
            if len(df.columns) == len(column_names):
                df = df.toDF(*column_names)
            else:
                print(f"Warning: Column count mismatch. Expected {len(column_names)}, got {len(df.columns)}. Skipping rename.")
        # Sanitize column names for Delta Lake compliance (no spaces, commas, etc.)
        import re
        for col_name in df.columns:
            # Replace any non-alphanumeric character with underscore
            new_name = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
            # Remove repeated underscores
            new_name = re.sub(r'_+', '_', new_name)
            # Remove leading/trailing underscores
            new_name = new_name.strip('_')
            
            if new_name != col_name:
                df = df.withColumnRenamed(col_name, new_name)

        if drop_columns:
            for col_to_drop in drop_columns:
                sanitized_drop = re.sub(r'[^a-zA-Z0-9]', '_', col_to_drop)
                sanitized_drop = re.sub(r'_+', '_', sanitized_drop).strip('_')
                if sanitized_drop in df.columns:
                    df = df.drop(sanitized_drop)
                    print(f"Dropped column {sanitized_drop} to prevent target leakage.")

        # Add metadata columns
        df_with_meta = df \
            .withColumn("_ingestion_timestamp", current_timestamp()) \
            .withColumn("_source_system", lit(source_system)) \
            .withColumn("_source_file", input_file_name()) \
            .withColumn("_dataset_name", lit(dataset_name))

        # Calculate schema hash
        schema_str = str(df.schema)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
        df_with_meta = df_with_meta.withColumn("_schema_hash", lit(schema_hash))

        # Write to Delta
        df_with_meta.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(output_path)
            
        print(f"Ingestion complete. Count: {df_with_meta.count()}")
        return df_with_meta
