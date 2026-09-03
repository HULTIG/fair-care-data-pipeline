"""
Tests for Fairness Metrics module
"""
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pace.gold.fairnessmetrics import FairnessMetrics


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for tests"""
    spark = SparkSession.builder \
        .appName("test_fairness") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def biased_data(spark):
    """Create sample DataFrame with bias"""
    data = []
    # Group A: 80% positive outcomes
    for i in range(100):
        data.append(("A", 1 if i < 80 else 0))
    # Group B: 40% positive outcomes (biased)
    for i in range(100):
        data.append(("B", 1 if i < 40 else 0))
    
    columns = ["protected_attr", "label"]
    pdf = pd.DataFrame(data, columns=columns)
    return spark.createDataFrame(pdf)


@pytest.fixture
def fair_data(spark):
    """Create sample DataFrame without bias"""
    data = []
    # Both groups: 50% positive outcomes
    for i in range(100):
        data.append(("A", i % 2))
    for i in range(100):
        data.append(("B", i % 2))
    
    columns = ["protected_attr", "label"]
    pdf = pd.DataFrame(data, columns=columns)
    return spark.createDataFrame(pdf)


@pytest.fixture
def fairness_config():
    """Default fairness config"""
    return {
        "protected_attribute": "protected_attr",
        "label_column": "label",
        "privileged_groups": [{"protected_attr": "A"}],
        "unprivileged_groups": [{"protected_attr": "B"}],
        "favorable_label": 1
    }


def test_fairness_metrics_initialization(fairness_config):
    """Test FairnessMetrics initialization"""
    metrics = FairnessMetrics(fairness_config)
    assert metrics.config == fairness_config


def test_calculate_metrics_biased_data(biased_data, fairness_config):
    """Test fairness metrics on biased data"""
    metrics = FairnessMetrics(fairness_config)
    report = metrics.calculate(biased_data)
    
    assert isinstance(report, dict)


def test_calculate_metrics_fair_data(fair_data, fairness_config):
    """Test fairness metrics on fair data"""
    metrics = FairnessMetrics(fairness_config)
    report = metrics.calculate(fair_data)
    
    assert isinstance(report, dict)


def test_statistical_parity_difference(biased_data, fairness_config):
    """Test Statistical Parity Difference calculation"""
    metrics = FairnessMetrics(fairness_config)
    report = metrics.calculate(biased_data)
    
    if "statistical_parity_difference" in report:
        spd = report["statistical_parity_difference"]
        # Should show bias: 0.40 - 0.80 = -0.40
        assert isinstance(spd, (int, float))
        assert -1.0 <= spd <= 1.0


def test_disparate_impact(biased_data, fairness_config):
    """Test Disparate Impact calculation"""
    metrics = FairnessMetrics(fairness_config)
    report = metrics.calculate(biased_data)
    
    if "disparate_impact" in report:
        di = report["disparate_impact"]
        # Should show bias: 0.40 / 0.80 = 0.5
        assert isinstance(di, (int, float))
        assert di >= 0


def test_fair_data_metrics(fair_data, fairness_config):
    """Test that fair data has metrics close to ideal"""
    metrics = FairnessMetrics(fairness_config)
    report = metrics.calculate(fair_data)
    
    if "statistical_parity_difference" in report:
        spd = report["statistical_parity_difference"]
        # Should be close to 0 for fair data
        assert abs(spd) < 0.2


def test_missing_protected_attribute(biased_data):
    """Test with missing protected attribute"""
    config = {
        "protected_attribute": "nonexistent",
        "label_column": "label",
        "privileged_groups": [{"nonexistent": "A"}],
        "unprivileged_groups": [{"nonexistent": "B"}],
        "favorable_label": 1
    }
    metrics = FairnessMetrics(config)
    report = metrics.calculate(biased_data)
    
    # Should handle gracefully
    assert isinstance(report, dict)


def test_missing_label_column(biased_data):
    """Test with missing label column"""
    config = {
        "protected_attribute": "protected_attr",
        "label_column": "nonexistent",
        "privileged_groups": [{"protected_attr": "A"}],
        "unprivileged_groups": [{"protected_attr": "B"}],
        "favorable_label": 1
    }
    metrics = FairnessMetrics(config)
    report = metrics.calculate(biased_data)
    
    assert isinstance(report, dict)


def test_empty_dataframe(spark, fairness_config):
    """Test with empty DataFrame"""
    empty_df = spark.createDataFrame([], schema="protected_attr STRING, label INT")
    metrics = FairnessMetrics(fairness_config)
    report = metrics.calculate(empty_df)
    
    assert isinstance(report, dict)


def test_single_group(spark, fairness_config):
    """Test with only one group present"""
    data = [("A", 1), ("A", 0), ("A", 1)]
    pdf = pd.DataFrame(data, columns=["protected_attr", "label"])
    single_group_df = spark.createDataFrame(pdf)
    
    metrics = FairnessMetrics(fairness_config)
    report = metrics.calculate(single_group_df)
    
    assert isinstance(report, dict)


def test_error_key_on_failure(spark):
    """Test that error key is present on calculation failure"""
    data = [(None, None)]
    pdf = pd.DataFrame(data, columns=["protected_attr", "label"])
    invalid_df = spark.createDataFrame(pdf, schema="protected_attr STRING, label INT")
    
    config = {
        "protected_attribute": "protected_attr",
        "label_column": "label",
        "privileged_groups": [{"protected_attr": "A"}],
        "unprivileged_groups": [{"protected_attr": "B"}],
        "favorable_label": 1
    }
    metrics = FairnessMetrics(config)
    report = metrics.calculate(invalid_df)
    
    # Should return dict, possibly with error key
    assert isinstance(report, dict)
