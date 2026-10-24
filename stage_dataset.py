"""Copy the audited YOLO dataset to a local SSD and verify every copied file."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Codex/smart-agriculture-cache/telepati-yolo"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stage(source: Path, target: Path, manifest: Path, workers: int) -> dict:
    source, target = source.resolve(), target.resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Source and target must be separate, non-nested directories.")
    source_yaml = source / "data.yaml"
    yaml_text = source_yaml.read_text(encoding="utf-8")
    staged_yaml, count = re.subn(r"(?m)^path:.*$", "path: " + json.dumps(target.as_posix()), yaml_text)
    if count != 1:
        raise ValueError("Expected exactly one top-level path entry in source data.yaml.")
    expected_images: dict[Path, str] = {}
    expected_labels: set[Path] = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["included"] != "1":
                continue
            image = Path(row["output_image"]).resolve().relative_to(source)
            label = Path(row["output_label"]).resolve().relative_to(source)
            if image.parts[0] != "images" or label.parts[0] != "labels":
                raise ValueError("Manifest file falls outside images/labels.")
            if image in expected_images or label in expected_labels:
                raise ValueError(f"Duplicate included manifest path: {image}")
            if not re.fullmatch(r"[0-9a-f]{64}", row["byte_sha256"]):
                raise ValueError(f"Invalid image SHA-256 in manifest: {image}")
            expected_images[image] = row["byte_sha256"]
            expected_labels.add(label)
    if not expected_images:
        raise ValueError("Manifest contains no included images.")
    for directory, expected in (("images", set(expected_images)), ("labels", expected_labels)):
        actual = {p.relative_to(source) for p in (source / directory).rglob("*") if p.is_file() and p.suffix != ".cache"}
        if actual != expected:
            raise ValueError(f"Source {directory} differ from included manifest (missing={len(expected-actual)}, extra={len(actual-expected)}).")
    tasks = [(relative, expected) for relative, expected in expected_images.items()]
    tasks += [(relative, None) for relative in expected_labels]
    tasks.append((Path("class_mapping.json"), None))
    allowed = {relative for relative, _ in tasks} | {Path("data.yaml")}
    if target.exists():
        unexpected = [p.relative_to(target) for p in target.rglob("*") if p.is_file() and p.suffix != ".cache" and p.relative_to(target) not in allowed]
        if unexpected:
            raise ValueError(f"Target has unexpected files: {unexpected[:5]}")
    target.mkdir(parents=True, exist_ok=True)
    for relative, _ in tasks:
        (target / relative).parent.mkdir(parents=True, exist_ok=True)

    def copy_verified(item: tuple[Path, str | None]) -> tuple[bool, int]:
        relative, expected = item
        original, destination = source / relative, target / relative
        if not original.is_file():
            raise ValueError(f"Source file missing: {original}")
        expected = expected or digest(original)
        copied = not destination.exists()
        if copied:
            temporary = destination.with_name(destination.name + ".staging-tmp")
            try:
                shutil.copy2(original, temporary)
                if digest(temporary) != expected:
                    raise ValueError(f"Copied bytes differ from expected SHA-256: {relative}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        elif not destination.is_file() or digest(destination) != expected:
            raise ValueError(f"Existing target file differs from source/manifest: {relative}")
        return copied, destination.stat().st_size

    copied, total_bytes = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_verified, item) for item in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            was_copied, size = future.result()
            copied += was_copied
            total_bytes += size
            if index % 1000 == 0 or index == len(tasks):
                print(f"Verified {index:,}/{len(tasks):,} files", flush=True)
    destination_yaml = target / "data.yaml"
    if destination_yaml.exists() and destination_yaml.read_text(encoding="utf-8") != staged_yaml:
        raise ValueError("Existing target data.yaml differs from the expected staged YAML.")
    destination_yaml.write_text(staged_yaml, encoding="utf-8")
    return dict(source=str(source), target=str(target), manifest=str(manifest.resolve()),
                manifest_sha256=digest(manifest), workers=workers,
                file_counts=dict(images=len(expected_images), labels=len(expected_labels), class_mapping=1,
                                 data_yaml=1, verified=len(tasks), copied=copied, already_identical=len(tasks)-copied),
                verified_bytes=total_bytes, verified_success=True, data_yaml=str(destination_yaml),
                verification="All images match included manifest byte_sha256; labels and class_mapping match source SHA-256. Source files are unchanged.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/yolo")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/manifest.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/ssd_staging.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8.")
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), source=str(args.source.resolve()),
                  target=str(args.target.resolve()), verified_success=False, status="failed")
    try:
        report.update(stage(args.source, args.target, args.manifest, args.workers))
        report["status"] = "complete"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["verified_success"]:
        print(f"Verified SSD dataset: {report['data_yaml']}", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
