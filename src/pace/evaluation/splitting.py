"""Stable source-record splitting for paired experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _membership_hash(values) -> str:
    encoded = "\n".join(sorted(str(value) for value in values)).encode()
    return hashlib.sha256(encoded).hexdigest()


def split_source_records(
    frame: pd.DataFrame,
    *,
    label_column: str,
    record_id_column: str = "_record_id",
    seed: int = 42,
    test_size: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create one stratified split from eligible original source records."""
    required = {label_column, record_id_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"source split is missing required columns: {missing}")
    eligible = frame.dropna(subset=[label_column, record_id_column]).copy()
    if eligible.empty or eligible[label_column].nunique() < 2:
        raise ValueError("source split requires eligible records with two outcome classes")
    if eligible[record_id_column].duplicated().any():
        raise ValueError("source split requires unique stable record identities")

    # Canonicalize input order so repeated reads with the same source IDs and
    # seed produce the same membership even if Spark returns rows differently.
    eligible = eligible.sort_values(record_id_column, key=lambda values: values.astype(str)).reset_index(drop=True)
    train_ids, test_ids = train_test_split(
        eligible[record_id_column].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=eligible[label_column].tolist(),
    )
    train = eligible[eligible[record_id_column].isin(train_ids)].copy()
    test = eligible[eligible[record_id_column].isin(test_ids)].copy()
    train_hash = _membership_hash(train[record_id_column])
    test_hash = _membership_hash(test[record_id_column])
    if set(train[record_id_column]).intersection(test[record_id_column]):
        raise ValueError("source split contains overlapping train/test identities")
    metadata = {
        "procedure": "stratified_train_test_split",
        "seed": int(seed),
        "test_fraction": float(test_size),
        "eligible_count": int(len(eligible)),
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "train_membership_sha256": train_hash,
        "test_membership_sha256": test_hash,
        "source_label_counts": {str(k): int(v) for k, v in eligible[label_column].value_counts().items()},
        "train_label_counts": {str(k): int(v) for k, v in train[label_column].value_counts().items()},
        "test_label_counts": {str(k): int(v) for k, v in test[label_column].value_counts().items()},
        "train_membership_ids": sorted(map(str, train[record_id_column])),
        "test_membership_ids": sorted(map(str, test[record_id_column])),
    }
    return train, test, metadata


def split_metadata_json(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True)
