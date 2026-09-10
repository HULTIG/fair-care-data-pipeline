from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit, input_file_name
import hashlib
import re
from pyspark.sql.functions import monotonically_increasing_id

class DataIngestion:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def ingest(self, source_path: str, output_path: str, dataset_name: str, source_system: str = "manual_upload", has_header: bool = True, column_names: list = None, delimiter: str = ",", drop_columns: list = None, predictor_allowlist: list = None, required_columns: list = None, missing_value_tokens: list = None, label_column: str = None, allowed_label_values: list = None, protected_attribute: str = None, required_group_values: list = None):
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
        for col_name in df.columns:
            # Replace any non-alphanumeric character with underscore
            new_name = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
            # Remove repeated underscores
            new_name = re.sub(r'_+', '_', new_name)
            # Remove leading/trailing underscores
            new_name = new_name.strip('_')
            
            if new_name != col_name:
                df = df.withColumnRenamed(col_name, new_name)

        normalized = {re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_') for name in (drop_columns or [])}
        if drop_columns:
            for col_to_drop in normalized:
                sanitized_drop = re.sub(r'[^a-zA-Z0-9]', '_', col_to_drop)
                sanitized_drop = re.sub(r'_+', '_', sanitized_drop).strip('_')
                if sanitized_drop in df.columns:
                    df = df.drop(sanitized_drop)
                    print(f"Dropped column {sanitized_drop} to prevent target leakage.")

        if missing_value_tokens:
            from pyspark.sql.functions import when, col
            for name in df.columns:
                df = df.withColumn(name, when(col(name).isin(missing_value_tokens), None).otherwise(col(name)))

        from pyspark.sql.functions import trim, col
        for name, dtype in df.dtypes:
            if dtype == "string":
                df = df.withColumn(name, trim(col(name)))

        if required_columns:
            required = {re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_') for name in required_columns}
            missing = sorted(required.difference(df.columns))
            if missing:
                raise ValueError(f"{dataset_name}: required columns are missing after normalization: {missing}")

        if predictor_allowlist:
            allowed = {re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_') for name in predictor_allowlist}
            required = {re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_') for name in (required_columns or [])}
            unexpected = sorted(set(df.columns) - allowed - required - {"_record_id"})
            if unexpected:
                df = df.drop(*unexpected)
                print(f"Dropped columns outside {dataset_name} predictor allowlist: {unexpected}")

        def observed(column_name):
            return {str(row[0]) for row in df.select(column_name).where(col(column_name).isNotNull()).distinct().collect()}
        if label_column and allowed_label_values:
            unknown = sorted(observed(label_column).difference({str(v) for v in allowed_label_values}))
            if unknown:
                raise ValueError(f"{dataset_name}: unknown labels in {label_column}: {unknown}")
        if protected_attribute and required_group_values:
            missing_groups = sorted({str(v) for v in required_group_values}.difference(observed(protected_attribute)))
            if missing_groups:
                raise ValueError(f"{dataset_name}: required comparison groups are absent: {missing_groups}")

        # This identity is internal alignment metadata and is never a predictor.
        df = df.withColumn("_record_id", monotonically_increasing_id())

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
