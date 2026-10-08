"""Verify the packaged CHALM scores against checksum receipts and raw metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def readable(path: Path) -> Path:
    path = path.resolve()
    return Path("\\\\?\\" + str(path)) if sys.platform == "win32" else path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with readable(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(*, package_only: bool = False, require_raw: bool = False) -> None:
    with (FINAL / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, "Empty metrics table"
    seeds = {row["training_seed"] for row in rows if row["agent"] == "chalm"}
    assert seeds == {"13", "29", "31"}, f"Unexpected CHALM seeds: {seeds}"
    source_map = json.loads((FINAL / "provenance" / "source_map.json").read_text(encoding="utf-8"))
    assert digest(FINAL / "metrics.csv") == source_map["metrics_sha256"]
    assert source_map["raw_jobs"] == 51
    assert source_map["displayed_seeds"] == [13, 29, 31]
    with (FINAL / "chalm_seed_summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 9, f"Expected nine CHALM summary rows, found {len(summary)}"
    for item in summary:
        scores = [
            100 * float(row[item["metric"]])
            for row in rows
            if row["agent"] == "chalm"
            and row["dataset"] == item["dataset"]
            and row["horizon"] == item["horizon"]
        ]
        assert len(scores) == int(item["n"]) == 3, item
        assert item["seeds"] == "13,29,31", item
        assert abs(statistics.mean(scores) - float(item["mean_percent"])) < 1e-7, item
        assert abs(statistics.stdev(scores) - float(item["sample_sd_percent"])) < 1e-7, item
    raw_jobs = FINAL / "raw" / "jobs"
    if require_raw:
        assert raw_jobs.is_dir(), f"Raw job archive missing: {raw_jobs}"
    if raw_jobs.is_dir() and not package_only:
        receipts = list(raw_jobs.rglob("complete.json"))
        assert len(receipts) == 51, f"Expected 51 completed jobs; found {len(receipts)}"
        for receipt_path in receipts:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for relative_name, expected in receipt.get(
                "release_artifact_sha256", receipt["artifacts"]
            ).items():
                artifact = receipt_path.parent / relative_name.replace("\\", "/")
                assert digest(artifact) == expected, f"Checksum mismatch: {artifact}"
        for row in rows:
            path = FINAL / row["metrics_file"]
            assert readable(path).is_file(), f"Missing metrics file: {path}"
        print(
            f"Verified {len(rows)} score rows and {len(receipts)} completed jobs; CHALM seeds 13, 29, 31"
        )
    else:
        print(
            f"Verified {len(rows)} packaged score rows and source-map hash; raw job audit skipped"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--package-only",
        action="store_true",
        help="Check the public summary without local raw jobs",
    )
    group.add_argument(
        "--require-raw", action="store_true", help="Require and checksum all raw job artifacts"
    )
    args = parser.parse_args()
    verify(package_only=args.package_only, require_raw=args.require_raw)
