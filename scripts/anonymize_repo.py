"""Audit anonymous releases; scrub metadata while preserving benchmark records by default."""

from pathlib import Path
import argparse, hashlib, json, re, shutil

TERMS = [
    "".join(map(chr, c))
    for c in [
        [82, 105, 121, 97, 110],
        [66, 104, 97, 114, 103, 97, 118, 97],
        [66, 73, 84, 83],
        [68, 117, 98, 97, 105],
    ]
]
PATTERN = re.compile(
    "|".join(map(re.escape, TERMS[:2])) + r"|\b(?:" + "|".join(map(re.escape, TERMS[2:])) + r")\b",
    re.I,
)
IGNORED = {".git", ".runtime", "__pycache__", ".venv"}
TEXT = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".md",
    ".txt",
    ".py",
    ".toml",
    ".ini",
    ".cfg",
    ".ps1",
    ".sh",
    ".bat",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scrub(value):
    if isinstance(value, str):
        return PATTERN.sub("anonymous", value)
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, tuple):
        return tuple(scrub(v) for v in value)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-dir", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.apply:
        if a.backup_dir is None:
            ap.error("--apply requires an external --backup-dir")
        backup = a.backup_dir.resolve()
        if backup == root or root in backup.parents:
            ap.error("Backup must be outside the release root")
        backup.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and not any(x in IGNORED for x in p.relative_to(root).parts)
    ]
    changes = []
    matches = []
    for p in files:
        rel = p.relative_to(root)
        benchmark = (
            p.name == "predictions.jsonl"
            or p.name.startswith("official_input_")
            or (rel.parts[0] == "data" and p.suffix in {".json", ".jsonl"})
        )
        if p.suffix in TEXT or p.name in [".gitignore", ".gitattributes"]:
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            hits = PATTERN.findall(text)
            if hits:
                matches.append(
                    dict(
                        path=str(rel),
                        matches=len(hits),
                        kind="benchmark" if benchmark else "metadata_or_source",
                    )
                )
                if a.apply and not benchmark:
                    target = backup / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError(
                            "Backup exists; choose a fresh backup directory: " + str(target)
                        )
                    shutil.copy2(p, target)
                    p.write_text(PATTERN.sub("anonymous", text), encoding="utf-8")
                    changes.append(dict(path=str(rel), before=digest(target), after=digest(p)))
        elif p.suffix == ".pt":
            import torch

            payload = torch.load(p, map_location="cpu", weights_only=False)
            strings = []

            def inspect(v):
                if isinstance(v, str):
                    strings.append(v)
                elif isinstance(v, dict):
                    for k, x in v.items():
                        inspect(k)
                        inspect(x)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        inspect(x)

            inspect(payload)
            if any(PATTERN.search(s) for s in strings):
                matches.append(
                    dict(
                        path=str(rel),
                        matches=sum(len(PATTERN.findall(s)) for s in strings),
                        kind="checkpoint_metadata",
                    )
                )
                if a.apply:
                    target = backup / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError(str(target))
                    shutil.copy2(p, target)
                    torch.save(scrub(payload), p)
                    changes.append(dict(path=str(rel), before=digest(target), after=digest(p)))
    if a.apply:
        # Record transformed release hashes, preserving original experimental fingerprints.
        for p in files:
            if p.name != "complete.json":
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            updated = {}
            for name, old in d.get("artifacts", {}).items():
                artifact = p.parent / name
                if artifact.is_file():
                    updated[name] = digest(artifact)
            if updated:
                d["release_artifact_sha256"] = updated
            if d.get("result"):
                pred = p.parent / d["result"] / "predictions.jsonl"
                if pred.exists():
                    d["release_predictions_sha256"] = digest(pred)
            p.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    if a.apply:
        manifest = root / "results/publication_manifest.json"
        if manifest.exists():
            d = json.loads(manifest.read_text(encoding="utf-8"))
            d["files"] = {
                str(p.relative_to(root)).replace("\\", "/"): digest(p)
                for p in (root / "results").rglob("*")
                if p.is_file() and p != manifest
            }
            d["format_version"] = 3
            manifest.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    result = dict(
        applied=a.apply,
        benchmark_redaction=False,
        matches=matches,
        changes=changes,
    )
    if a.report:
        a.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                applied=a.apply,
                audited_files=len(files),
                matching_files=len(matches),
                benchmark_files=sum(x["kind"] == "benchmark" for x in matches),
                metadata_files=sum(x["kind"] != "benchmark" for x in matches),
                changed_files=len(changes),
            ),
            indent=2,
        )
    )
    if matches and not a.apply:
        if any(x["kind"] != "benchmark" for x in matches):
            print("Source or metadata matches remain; anonymization is incomplete.")
        else:
            print(
                "Only original benchmark text matches remain; source and metadata have no matches."
            )


if __name__ == "__main__":
    main()
