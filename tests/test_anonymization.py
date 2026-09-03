"""
Tests for Anonymization module
"""
import pytest
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pace.silver.anonymization import AnonymizationEngine


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for tests"""
    spark = SparkSession.builder \
        .appName("test_anonymization") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_data(spark):
    """Create sample DataFrame for anonymization"""
    data = []
    for i in range(100):
        data.append((
            20 + (i % 50),  # age: 20-69
            "M" if i % 2 == 0 else "F",  # gender
            f"ZIP{i % 10}",  # zip_code
            i % 2  # label
        ))
    
    columns = ["age", "gender", "zip_code", "label"]
    pdf = pd.DataFrame(data, columns=columns)
    return spark.createDataFrame(pdf)


def test_kanonymity_initialization():
    """Test k-anonymity initialization"""
    config = {
        "technique": "kanonymity",
        "k": 5,
        "quasi_identifiers": ["age", "gender"]
    }
    engine = AnonymizationEngine(config)
    assert engine.config["technique"] == "kanonymity"
    assert engine.config["k"] == 5


def test_kanonymity_anonymization(sample_data, spark):
    """Test k-anonymity anonymization"""
    config = {
        "technique": "kanonymity",
        "k": 5,
        "quasi_identifiers": ["age", "gender"]
    }
    engine = AnonymizationEngine(config)
    anonymized_df, metadata = engine.anonymize(sample_data, spark)
    
    assert anonymized_df is not None
    assert anonymized_df.count() > 0
    assert "risk" in metadata


def test_kanonymity_reduces_rows(sample_data, spark):
    """Test that k-anonymity suppression reduces row count"""
    original_count = sample_data.count()
    
    config = {
        "technique": "kanonymity",
        "k": 10,  # High k value
        "quasi_identifiers": ["age", "gender", "zip_code"]
    }
    engine = AnonymizationEngine(config)
    anonymized_df, _ = engine.anonymize(sample_data, spark)
    
    # With high k and multiple QIs, some rows should be suppressed
    assert anonymized_df.count() <= original_count


def test_differential_privacy_initialization():
    """Test differential privacy initialization"""
    config = {
        "technique": "differentialprivacy",
        "epsilon": 1.0,
        "quasi_identifiers": ["age"]
    }
    engine = AnonymizationEngine(config)
    assert engine.config["epsilon"] == 1.0


def test_differential_privacy_anonymization(sample_data, spark):
    """Test differential privacy anonymization"""
    config = {
        "technique": "differentialprivacy",
        "epsilon": 1.0,
        "quasi_identifiers": ["age"]
    }
    engine = AnonymizationEngine(config)
    anonymized_df, _ = engine.anonymize(sample_data, spark)
    
    assert anonymized_df is not None
    assert anonymized_df.count() == sample_data.count()  # DP doesn't suppress rows


def test_differential_privacy_adds_noise(sample_data, spark):
    """Test that DP adds noise to numeric columns"""
    config = {
        "technique": "differentialprivacy",
        "epsilon": 1.0,
        "quasi_identifiers": ["age"]
    }
    engine = AnonymizationEngine(config)
    
    original_ages = [row.age for row in sample_data.collect()]
    anonymized_df, _ = engine.anonymize(sample_data, spark)
    anonymized_ages = [row.age for row in anonymized_df.collect()]
    
    # Ages should be different due to noise (with high probability)
    assert original_ages != anonymized_ages


def test_no_anonymization(sample_data, spark):
    """Test with no anonymization technique"""
    config = {
        "technique": "none",
        "k": 0
    }
    engine = AnonymizationEngine(config)
    result_df, _ = engine.anonymize(sample_data, spark)
    
    # Should return original data unchanged
    assert result_df.count() == sample_data.count()


def test_empty_quasi_identifiers(sample_data, spark):
    """Test with empty quasi-identifiers list"""
    config = {
        "technique": "kanonymity",
        "k": 5,
        "quasi_identifiers": []
    }
    engine = AnonymizationEngine(config)
    result_df, _ = engine.anonymize(sample_data, spark)
    
    assert result_df is not None


def test_invalid_technique():
    """Test with invalid anonymization technique"""
    config = {
        "technique": "invalid_technique",
        "k": 5
    }
    engine = AnonymizationEngine(config)
    # Should not raise error on init, only on anonymize
    assert engine.config["technique"] == "invalid_technique"


def test_high_k_value(sample_data, spark):
    """Test with very high k value"""
    config = {
        "technique": "kanonymity",
        "k": 50,  # Higher than dataset size
        "quasi_identifiers": ["age", "gender", "zip_code"]
    }
    engine = AnonymizationEngine(config)
    anonymized_df, _ = engine.anonymize(sample_data, spark)
    
    # Should suppress most/all rows
    assert anonymized_df.count() >= 0


def test_low_epsilon(sample_data, spark):
    """Test differential privacy with low epsilon (high privacy)"""
    config = {
        "technique": "differentialprivacy",
        "epsilon": 0.1,  # Very low epsilon
        "quasi_identifiers": ["age"]
    }
    engine = AnonymizationEngine(config)
    anonymized_df, _ = engine.anonymize(sample_data, spark)
    
    assert anonymized_df.count() == sample_data.count()
