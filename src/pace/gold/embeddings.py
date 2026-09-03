from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer
import pandas as pd

class EmbeddingsGenerator:
    def __init__(self, config: dict):
        self.config = config
        # Initialize model (lazy load might be better in distributed setting, but ok for local)
        self.model_name = "all-MiniLM-L6-v2"

    def generate(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Generates embeddings for text columns.
        """
        print("Generating Embeddings...")
        text_cols = self.config.get("text_columns", [])
        
        if not text_cols:
            # Try to infer text columns or skip
            print("No text columns configured for embeddings.")
            return df
            
        # Using pandas UDF or mapPartitions is better for performance with heavy models
        # Here we use a simplified approach: collect to pandas, embed, create DF
        # WARNING: Not scalable for huge data, but fits the 'local docker' scope
        
        pdf = df.toPandas()
        model = SentenceTransformer(self.model_name)
        
        for col in text_cols:
            if col in pdf.columns:
                print(f"Embedding column: {col}")
                embeddings = model.encode(pdf[col].astype(str).tolist())
                pdf[f"{col}_embedding"] = list(embeddings)
                
        return spark.createDataFrame(pdf)
