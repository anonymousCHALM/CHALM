"""Verify published CHALM result bytes and completed evaluation coverage."""

from pathlib import Path
import csv, json, hashlib, yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    d = json.loads((ROOT / "results/publication_manifest.json").read_text())
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "results").rglob("*")
        if p.is_file()
        and p.name != "publication_manifest.json"
        and not any(x in p.parts for x in ["reproduction", "smoke", "__pycache__"])
    }
    assert actual == set(d["files"]), "Result inventory differs from the publication manifest"
    for rel, expected in d["files"].items():
        p = ROOT / rel
        if not p.is_file() or digest(p) != expected:
            raise ValueError(f"Missing/changed file: {rel}; run git lfs pull if needed")
    config = yaml.safe_load((ROOT / "experiments/evaluation_study.yaml").read_text())
    assert sorted(config["checkpoints"]) == [13, 29, 31]
    provenance = json.loads((ROOT / "results/final/provenance/run_manifest.json").read_text())
    for seed, path in config["checkpoints"].items():
        assert (ROOT / path).is_file()
        assert digest(ROOT / path) == provenance["release_checkpoint_sha256"][str(seed)]
    for suite, n in [("evaluation_study", 2), ("followup_study", 102)]:
        receipts = list((ROOT / "results" / suite / "jobs").rglob("complete.json"))
        assert len(receipts) == n
        for p in receipts:
            receipt = json.loads(p.read_text())
            run = p.parent / receipt["result"]
            assert (run / "predictions.jsonl").is_file() and (run / "config.yaml").is_file()
    assert len(list((ROOT / "results/final/raw/jobs").rglob("complete.json"))) == 51
    rows = list(
        csv.DictReader((ROOT / "results/followup_study/significance/paired_tests.csv").open())
    )
    assert len(rows) == 20 and all(x["complete"] == "True" for x in rows)
    assert all(x["p_holm"] == "" for x in rows if x["training_seed"] != "mean")
    print(
        "Verified",
        len(d["files"]),
        "published result files, three final checkpoints, 155 completed jobs and 20 requested follow-up test rows.",
    )


if __name__ == "__main__":
    main()
