import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "summarize_results.py"
spec = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def samples(*lines):
    return [{"whole_device_samples_index_mib_util_percent_watts": line} for line in lines]


def test_unsupported_power_is_missing_but_other_readings_are_preserved():
    (row,) = reporting.device_sample_summary(samples("0, 1058, 2, [N/A]"))
    assert row["memory_mib_mean"] == 1058
    assert row["utilization_percent_max"] == 2
    assert row["power_watts_samples"] == 0
    assert "power_watts_mean" not in row
    assert "power_watts_max" not in row


def test_mixed_invalid_readings_do_not_contaminate_finite_samples():
    rows = reporting.device_sample_summary(
        samples(
            "0, 100, 0, 0\n1, 200, 25, 40",
            "0, 300, nan, [N/A]\n1, 400, 75, inf",
            "malformed",
        )
    )
    a, b = rows
    assert a["device"] == "0" and b["device"] == "1"
    assert a["memory_mib_mean"] == pytest.approx(200)
    assert a["utilization_percent_samples"] == 1
    assert a["power_watts_mean"] == 0
    assert b["utilization_percent_mean"] == pytest.approx(50)
    assert b["power_watts_samples"] == 1
    assert b["power_watts_max"] == 40


def test_absent_device_samples_have_no_invented_device():
    assert reporting.device_sample_summary([{}]) == []
