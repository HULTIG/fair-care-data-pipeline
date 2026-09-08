"""
Tests for Bias Mitigation module
"""
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pace.gold.biasmitigation import BiasMitigator


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for tests"""
    spark = SparkSession.builder \
        .appName("test_biasmitigation") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()


def _adult_style_df(spark):
    """Adult-style data: string labels with leading whitespace (raw CSV form)."""
    data = []
    for i in range(100):
        data.append(("Male" if i % 2 == 0 else "Female", " >50K" if i < 40 else " <=50K"))
    pdf = pd.DataFrame(data, columns=["sex", "income"])
    return spark.createDataFrame(pdf)


def _adult_style_config():
    return {
        "protected_attribute": "sex",
        "label_column": "income",
        "privileged_groups": [{"sex": "Male"}],
        "unprivileged_groups": [{"sex": "Female"}],
        "favorable_label": ">50K",
    }


def _german_style_df(spark):
    """German-style data: numeric labels {1, 2} instead of {0, 1}."""
    data = []
    for i in range(100):
        data.append(("A91" if i % 2 == 0 else "A92", 1 if i < 50 else 2))
    pdf = pd.DataFrame(data, columns=["personal_status_sex", "credit_risk"])
    return spark.createDataFrame(pdf)


def _german_style_config():
    return {
        "protected_attribute": "personal_status_sex",
        "label_column": "credit_risk",
        "privileged_groups": [{"personal_status_sex": "A91"}],
        "unprivileged_groups": [{"personal_status_sex": "A92"}],
        "favorable_label": 1,
    }


def test_mitigate_preserves_original_string_labels(spark):
    """Mitigation must return original label values, not label-encoded ints.

    Downstream fairness maps config string labels ('>50K') onto the frame;
    returning encoded ints breaks that mapping (adult exp failure).
    """
    out = BiasMitigator(_adult_style_config()).mitigate(_adult_style_df(spark), spark)
    pdf = out.toPandas()

    assert "instance_weights" in pdf.columns
    values = set(pdf["income"].astype(str).str.strip().unique())
    assert values == {">50K", "<=50K"}, f"labels were not restored, got {values}"


def test_mitigate_non_zero_one_labels(spark):
    """Regression: german-style {1, 2} labels must be reweighed, not skipped."""
    out = BiasMitigator(_german_style_config()).mitigate(_german_style_df(spark), spark)
    pdf = out.toPandas()

    assert "instance_weights" in pdf.columns
    assert set(pdf["credit_risk"].unique()) == {1, 2}


def test_nij_config_groups_match_raw_data():
    """Regression: nij config Race groups must match actual data values.

    The config previously used 'White'/'Black' while the data contains
    'WHITE'/'BLACK', so fairness silently fell back to defaults (NaN groups).
    """
    from pathlib import Path

    import pandas as pd
    import yaml

    repo_root = Path(__file__).parent.parent
    with open(repo_root / "configs/default.yaml") as f:
        config = yaml.safe_load(f)
    nij = config["datasets"]["nij"]

    raw = pd.read_csv(repo_root / nij["raw_path"], usecols=["Race"])
    actual = set(raw["Race"].astype(str).str.strip().unique())
    configured = {
        list(g.values())[0]
        for g in nij["privileged_groups"] + nij["unprivileged_groups"]
    }
    assert configured <= actual, f"config groups {configured} not in data {actual}"
