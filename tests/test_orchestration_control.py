import json

import pytest

from pace.orchestration.pipeline import run_pipeline


def test_failed_run_writes_failure_record_without_summary(tmp_path):
    with pytest.raises(ValueError):
        run_pipeline("missing", {"datasets": {}}, str(tmp_path), seed=7)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    failure = run_dirs[0] / "failure.json"
    assert failure.exists()
    payload = json.loads(failure.read_text())
    assert payload["status"] == "failed"
    assert not list(run_dirs[0].glob("*_metricssummary.json"))
