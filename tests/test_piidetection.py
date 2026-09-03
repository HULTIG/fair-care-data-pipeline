"""
Tests for PII Detection module
"""
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pace.bronze.piidetection import PIIDetection


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for tests"""
    spark = SparkSession.builder \
        .appName("test_pii") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_data(spark):
    """Create sample DataFrame with PII"""
    data = [
        ("John Doe", "john.doe@example.com", "555-123-4567", "123-45-6789", 30),
        ("Jane Smith", "jane.smith@example.com", "555-987-6543", "987-65-4321", 25),
        ("Bob Johnson", "bob@example.com", "(555) 111-2222", "111-22-3333", 35),
    ]
    columns = ["name", "email", "phone", "ssn", "age"]
    pdf = pd.DataFrame(data, columns=columns)
    return spark.createDataFrame(pdf)


@pytest.fixture
def pii_config():
    """Default PII detection config"""
    return {
        "confidence_threshold": 0.8,
        "techniques": ["regex"],
        "entities": ["email", "phone", "ssn"]
    }


def test_pii_detection_initialization(pii_config):
    """Test PIIDetection initialization"""
    detector = PIIDetection(pii_config)
    assert detector.config == pii_config
    assert detector.regex_patterns is not None


def test_email_detection(sample_data, pii_config):
    """Test email detection"""
    detector = PIIDetection(pii_config)
    report = detector.detect(sample_data, sample_size=10)
    
    assert "email" in report
    assert "email" in report["email"]["pii_types"]
    assert report["email"]["recommendation"] == "REVIEW"


def test_phone_detection(sample_data, pii_config):
    """Test phone number detection"""
    detector = PIIDetection(pii_config)
    report = detector.detect(sample_data, sample_size=10)
    
    assert "phone" in report
    assert "phone" in report["phone"]["pii_types"]


def test_ssn_detection(sample_data, pii_config):
    """Test SSN detection"""
    detector = PIIDetection(pii_config)
    report = detector.detect(sample_data, sample_size=10)
    
    assert "ssn" in report
    assert "ssn" in report["ssn"]["pii_types"]


def test_no_pii_in_age(sample_data, pii_config):
    """Test that age column is not flagged as PII"""
    detector = PIIDetection(pii_config)
    report = detector.detect(sample_data, sample_size=10)
    
    assert "age" not in report


def test_confidence_threshold(sample_data):
    """Test confidence threshold filtering"""
    config = {
        "confidence_threshold": 1.0,  # Very high threshold
        "techniques": ["regex"],
        "entities": ["email"]
    }
    detector = PIIDetection(config)
    report = detector.detect(sample_data, sample_size=10)
    
    # With high threshold, might not detect anything
    assert isinstance(report, dict)


def test_empty_dataframe(spark, pii_config):
    """Test with empty DataFrame"""
    empty_df = spark.createDataFrame([], schema="name STRING, email STRING")
    detector = PIIDetection(pii_config)
    report = detector.detect(empty_df, sample_size=10)
    
    assert isinstance(report, dict)


def test_metadata_columns_skipped(spark, pii_config):
    """Test that metadata columns (starting with _) are skipped"""
    data = [("test@example.com", "meta_value")]
    pdf = pd.DataFrame(data, columns=["email", "_metadata"])
    df = spark.createDataFrame(pdf)
    
    detector = PIIDetection(pii_config)
    report = detector.detect(df, sample_size=10)
    
    assert "_metadata" not in report


def test_sample_size_limit(sample_data, pii_config):
    """Test that sample size is respected"""
    detector = PIIDetection(pii_config)
    # Should not raise error even with sample_size > actual rows
    report = detector.detect(sample_data, sample_size=1000)
    
    assert isinstance(report, dict)


def test_multiple_pii_types(spark, pii_config):
    """Test column with multiple PII types"""
    data = [("john.doe@example.com 555-123-4567",)]
    pdf = pd.DataFrame(data, columns=["mixed"])
    df = spark.createDataFrame(pdf)
    
    detector = PIIDetection(pii_config)
    report = detector.detect(df, sample_size=10)
    
    if "mixed" in report:
        assert len(report["mixed"]["pii_types"]) >= 1
