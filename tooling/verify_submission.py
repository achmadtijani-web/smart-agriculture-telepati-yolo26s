"""Verify a completed public submission without importing Torch or using a GPU.

This checks artifacts and provenance, not the scientific validity of the labels.
Run after packaging: python tooling/verify_submission.py --root <project-folder>
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from urllib.parse import unquote, urlsplit
import zipfile


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {".venv", "venv", "__pycache__", ".git", "node_modules", "raw"}
NOTEBOOK_NAME = "Smart_Agriculture_Experiments.ipynb"
REQUIRED_REPOSITORY = (
    "train.py", "evaluate.py", "predict.py", "runtime_config.py",
    "prepare_dataset.py", "stage_dataset.py", "requirements.txt",
    "README.md", "MODEL_CARD.md", ".gitignore", NOTEBOOK_NAME,
)
REQUIRED_ASSETS = ("best.pt", NOTEBOOK_NAME, "metrics.json", "MODEL_CARD.md", "SHA256SUMS.txt")
METRIC_KEYS = ("mAP50", "mAP50_95", "macro_precision_iou50", "macro_recall_iou50", "macro_f1_iou50")
CLASS_KEYS = ("precision_iou50", "recall_iou50", "f1_iou50", "ap50", "ap50_95")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def private_path_present(payload: bytes) -> bool:
    """Find the original machine paths even when JSON or a pickle escapes slashes."""
    fragments = ("d:" + "\\" + "computer vision", "c:" + "\\" + "users" + "\\" + "t7099")
    for encoding in ("utf-8", "utf-16-le"):
        text = payload.decode(encoding, errors="ignore").casefold().replace("/", "\\")
        for _ in range(4):
            text = text.replace("\\\\", "\\")
        if any(fragment in text for fragment in fragments):
            return True
    return False


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and not path.is_absolute() and ":" not in name and ".." not in path.parts


class Verification:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.package = self.root / "package"
        self.repository = self.package / "repository"
        self.assets = self.package / "release_assets"
        self.manifest_path = self.root / "submission_manifest.json"
        self.checks: list[dict] = []
        self.details: dict = {}

    def check(self, name: str, condition: bool, detail=None) -> bool:
        item = {"check": name, "passed": bool(condition)}
        if detail is not None:
            item["detail"] = detail
        self.checks.append(item)
        return bool(condition)

    def guarded(self, name: str, function):
        try:
            return function()
        except Exception as exc:
            self.check(name, False, f"{type(exc).__name__}: {exc}")
            return None

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def required_files(self):
        missing_repo = [name for name in REQUIRED_REPOSITORY if not (self.repository / name).is_file()]
        missing_assets = [name for name in REQUIRED_ASSETS if not (self.assets / name).is_file()]
        self.check("repository_required_files", not missing_repo, {"missing": missing_repo})
        self.check("release_required_files", not missing_assets, {"missing": missing_assets})
        self.check("submission_manifest_exists", self.manifest_path.is_file())
        self.check("repository_docs_directory", (self.repository / "docs").is_dir())
        for label, folder in (("repository", self.repository), ("release_assets", self.assets)):
            prohibited = []
            symlinks = []
            if folder.is_dir():
                for path in folder.rglob("*"):
                    relative = path.relative_to(folder)
                    if any(part.casefold() in FORBIDDEN_PARTS for part in relative.parts):
                        prohibited.append(relative.as_posix())
                    if path.suffix.casefold() in {".pyc", ".pyo", ".tmp", ".bak", ".log"}:
                        prohibited.append(relative.as_posix())
                    if path.is_symlink():
                        symlinks.append(relative.as_posix())
            self.check(f"{label}_no_private_runtime_or_raw_data", not prohibited, prohibited[:30])
            self.check(f"{label}_no_symlinks", not symlinks, symlinks[:30])

    def public_privacy(self):
        inspected = 0
        violations = []
        for folder in (self.repository, self.assets):
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                inspected += 1
                if path.suffix.casefold() in {".pt", ".pth"} and zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as archive:
                        for item in archive.infolist():
                            # Tensor storage does not contain train_args; metadata is in data.pkl.
                            if item.filename.endswith((".pkl", ".json", ".txt")):
                                if private_path_present(archive.read(item)):
                                    violations.append(self.relative(path) + "::" + item.filename)
                elif private_path_present(path.read_bytes()):
                    violations.append(self.relative(path))
        self.check("public_files_no_original_private_paths", inspected > 0 and not violations,
                   {"files_inspected": inspected, "violations": violations})

    def checkpoint_containers(self):
        """Check the modern Torch container without unpickling model code."""
        paths = sorted(self.assets.rglob("*.pt"))
        failures = []
        structures = []
        for path in paths:
            if not zipfile.is_zipfile(path):
                failures.append({"file": self.relative(path), "error": "Not a modern Torch ZIP checkpoint"})
                continue
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                names = [item.filename for item in members if not item.is_dir()]
                metadata = [name for name in names if name.endswith("/data.pkl")]
                storages = [item for item in members if re.search(r"/data/[^/]+$", item.filename) and item.file_size > 0]
                if len(metadata) != 1 or not storages or any(not safe_member(name) for name in names) or len(names) != len(set(names)):
                    failures.append({"file": self.relative(path), "error": "Missing metadata/tensor storage, unsafe members, or duplicate entries"})
                structures.append({"file": self.relative(path), "tensor_storage_members": len(storages),
                                   "serialized_tensor_bytes": sum(item.file_size for item in storages)})
        self.check("release_torch_checkpoint_containers", bool(paths) and not failures,
                   {"files": structures, "failures": failures,
                    "limit": "Container structure only. Tensor identity and loading are separately checked by the CPU publication sanitizer."})

    def python_sources(self):
        paths = sorted(self.repository.rglob("*.py"))
        failures = []
        for path in paths:
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.name)
            except (SyntaxError, UnicodeError) as exc:
                failures.append({"file": self.relative(path), "error": str(exc)})
        self.check("repository_python_ast", bool(paths) and not failures,
                   {"count": len(paths), "failures": failures})

    def notebook(self):
        source_path = self.repository / NOTEBOOK_NAME
        asset_path = self.assets / NOTEBOOK_NAME
        notebook = read_json(source_path)
        self.check("notebook_identical_in_source_and_release", digest(source_path) == digest(asset_path))
        self.check("notebook_format", notebook.get("nbformat") == 4 and isinstance(notebook.get("cells"), list))
        code_count, output_count, non_null_count = 0, 0, 0
        output_ids = []
        failures, magics = [], []
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            code_count += 1
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            output_count += len(cell.get("outputs", []))
            if cell.get("outputs"):
                output_ids.append(cell.get("id"))
            non_null_count += int(cell.get("execution_count") is not None)
            lines = source.splitlines(keepends=True)
            first_nonempty = next((line.strip() for line in lines if line.strip()), "")
            if first_nonempty.startswith("%%"):
                magic = first_nonempty.split()[0][2:]
                magics.append({"cell": index, "magic": magic})
                if magic in {"capture", "time", "timeit"}:
                    first_index = next(i for i, line in enumerate(lines) if line.strip())
                    lines[first_index] = "\n"
                else:
                    failures.append({"cell": index, "error": "Non-Python cell magic requires a separate review: " + magic})
                    continue
            transformed = []
            for line in lines:
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]
                # These are IPython shell/line magics, not Python syntax errors.
                transformed.append(indent + "pass\n" if stripped.startswith(("!", "%")) else line)
            try:
                ast.parse("".join(transformed), filename=f"{NOTEBOOK_NAME}:cell{index}")
            except (SyntaxError, TypeError) as exc:
                failures.append({"cell": index, "error": str(exc)})
        self.check("notebook_code_ast", code_count > 0 and not failures,
                   {"code_cells": code_count, "failures": failures, "cell_magics": magics})
        provenance = notebook.get("metadata", {}).get("execution_provenance")
        if output_count or non_null_count:
            honest = isinstance(provenance, dict) and bool(provenance.get("description")) and bool(provenance.get("source"))
            self.check("notebook_execution_provenance", honest,
                       {"outputs": output_count, "executed_cells": non_null_count,
                        "provenance": provenance, "limit": "Recorded provenance cannot independently prove an execution occurred."})
            if honest:
                source = provenance["source"]
                source_path = self.repository / source
                self.check("notebook_provenance_builder_source", safe_member(source) and source_path.is_file()
                           and provenance.get("source_sha256") == digest(source_path))
                self.check("notebook_cpu_execution_scope", provenance.get("model_operations_executed") is False
                           and provenance.get("recorded_cpu_analysis_cells") == len(output_ids)
                           and provenance.get("recorded_output_cell_ids") == output_ids
                           and provenance.get("safe_default_code_cells_executed") == code_count)
                evidence = provenance.get("input_artifacts", [])
                invalid = []
                for item in evidence:
                    name = item.get("path", "")
                    path = self.repository / name
                    if not safe_member(name) or not path.is_file() or item.get("sha256") != digest(path):
                        invalid.append(name)
                self.check("notebook_provenance_input_hashes", bool(evidence) and not invalid,
                           {"files": len(evidence), "invalid": invalid})
        else:
            self.check("notebook_execution_provenance", True,
                       {"outputs": 0, "executed_cells": 0, "policy": "Unexecuted notebook; no fabricated execution outputs."})

    def evaluation_metrics(self):
        metrics = read_json(self.assets / "metrics.json")
        manifest = read_json(self.manifest_path)
        expected_weights = digest(self.assets / "best.pt")
        self.check("release_weights_match_evaluation_sha256", metrics.get("weights_sha256") == expected_weights)
        self.check("manifest_weights_match_release_sha256", manifest.get("weights_sha256") == expected_weights)
        self.check("selected_model_named", isinstance(manifest.get("selected_model"), str) and bool(manifest["selected_model"]))
        for split, gt in (("val", 4758), ("test", 2669)):
            values = metrics.get(split, {})
            self.check(f"{split}_metric_range", all(is_number(values.get(key)) and 0 <= values[key] <= 1 for key in METRIC_KEYS))
            self.check(f"{split}_class_and_gt_counts", values.get("classes_evaluated") == 18 and values.get("gt_instances") == gt,
                       {"classes": values.get("classes_evaluated"), "gt_instances": values.get("gt_instances")})
        threshold = metrics.get("selected_confidence")
        self.check("confidence_range", is_number(threshold) and 0.001 <= threshold <= 1)
        self.check("validation_threshold_policy_documented", "val" in str(metrics.get("threshold_selection", "")).casefold())
        self.check("evaluation_ap_protocol", metrics.get("ap_confidence_floor") == 0.001 and metrics.get("nms_iou") == 0.7 and metrics.get("max_det") == 300)
        forms = manifest.get("submission", {})
        expected = {"mAP50_percent": 100 * metrics["test"]["mAP50"],
                    "F1_percent": 100 * metrics["test"]["macro_f1_iou50"]}
        for key, value in expected.items():
            actual = forms.get(key)
            # A form may use two decimal places. A difference >0.0051 is not rounding.
            self.check("submission_" + key, is_number(actual) and 0 <= actual <= 100 and abs(actual - value) <= 0.0051,
                       {"manifest_percent": actual, "expected_full_precision_percent": value, "rounding_tolerance": 0.0051})
        self.details["submission"] = expected
        self.details["weights_sha256"] = expected_weights
        metrics_file = manifest.get("metrics_file")
        metrics_path = self.root / str(metrics_file) if metrics_file else None
        self.check("manifest_metrics_file_matches_release", metrics_path is not None and metrics_path.is_file() and digest(metrics_path) == digest(self.assets / "metrics.json"))
        repo_metrics = list(self.repository.rglob("metrics.json"))
        self.check("repository_contains_exact_release_metrics", any(digest(path) == digest(self.assets / "metrics.json") for path in repo_metrics))
        self.guarded("per_class_metrics", lambda: self.per_class(metrics))

    def per_class(self, metrics):
        mapping_paths = list(self.repository.rglob("class_mapping.json"))
        self.check("class_mapping_in_repository", bool(mapping_paths))
        for path in mapping_paths:
            mapping = read_json(path)
            rows = mapping.get("classes", [])
            ids = [row.get("yolo_id") for row in rows]
            self.check("class_mapping_18_" + self.relative(path), mapping.get("active_class_count") == 18 and sorted(ids) == list(range(18)) and len({row.get("name") for row in rows}) == 18)
        found = list(self.repository.rglob("per_class_metrics.csv"))
        self.check("per_class_metrics_csv_present", bool(found))
        for path in found:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for split in ("val", "test"):
                selected = [row for row in rows if row.get("split") == split]
                self.check(f"class_ids_{split}_" + self.relative(path), sorted(int(row["class_id"]) for row in selected) == list(range(18)))
                self.check(f"class_metric_ranges_{split}_" + self.relative(path), all(0 <= float(row[key]) <= 1 for row in selected for key in CLASS_KEYS))
                if len(selected) != 18:
                    continue
                for csv_key, json_key in (("precision_iou50", "macro_precision_iou50"), ("recall_iou50", "macro_recall_iou50"), ("f1_iou50", "macro_f1_iou50"), ("ap50", "mAP50"), ("ap50_95", "mAP50_95")):
                    mean = sum(float(row[csv_key]) for row in selected) / 18
                    self.check(f"macro_{split}_{csv_key}_" + self.relative(path), abs(mean - metrics[split][json_key]) <= 1e-9)
                self.check(f"gt_count_{split}_" + self.relative(path), sum(int(row["gt_instances"]) for row in selected) == metrics[split]["gt_instances"])
                self.check(f"class_confidence_{split}_" + self.relative(path), all(abs(float(row["confidence"]) - metrics["selected_confidence"]) <= 1e-12 for row in selected))
                self.check(f"class_f1_formula_{split}_" + self.relative(path), all(abs(float(row["f1_iou50"]) - 2 * float(row["precision_iou50"]) * float(row["recall_iou50"]) / (float(row["precision_iou50"]) + float(row["recall_iou50"]) + 1e-16)) <= 1e-9 for row in selected))

    def checksum_file(self):
        path = self.assets / "SHA256SUMS.txt"
        entries = {}
        errors = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line.strip())
            if not match:
                errors.append("Invalid checksum line")
                continue
            value, name = match.groups()
            if not safe_member(name) or name in entries or name == "SHA256SUMS.txt":
                errors.append("Unsafe, duplicate, or recursive checksum entry: " + name)
                continue
            entries[name] = value.casefold()
        expected = {path.relative_to(self.assets).as_posix(): path for path in self.assets.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"}
        self.check("release_checksum_file_members", not errors and set(entries) == set(expected),
                   {"errors": errors, "missing": sorted(set(expected) - set(entries)), "unexpected": sorted(set(entries) - set(expected))})
        wrong = [name for name in entries if name not in expected or digest(expected[name]) != entries[name]]
        self.check("release_checksum_file_hashes", bool(entries) and not wrong, wrong)

    def manifest_hashes(self):
        manifest = read_json(self.manifest_path)
        files = manifest.get("files")
        self.check("manifest_file_inventory", isinstance(files, dict) and bool(files))
        if not isinstance(files, dict):
            return
        wrong = []
        for name, descriptor in files.items():
            if not safe_member(name):
                wrong.append({"file": name, "error": "unsafe relative path"})
                continue
            path = self.root / name
            if not path.is_file():
                wrong.append({"file": name, "error": "missing"})
                continue
            expected_hash = descriptor.get("sha256") if isinstance(descriptor, dict) else descriptor
            expected_size = descriptor.get("bytes") if isinstance(descriptor, dict) else None
            if expected_hash != digest(path) or (expected_size is not None and expected_size != path.stat().st_size):
                wrong.append({"file": name, "error": "hash or byte size mismatch"})
        critical = {self.relative(self.assets / name) for name in REQUIRED_ASSETS}
        critical.update({"package/smart-agriculture-source.zip", "package/smart-agriculture-release.zip"})
        self.check("manifest_critical_files_covered", critical <= set(files), {"missing": sorted(critical - set(files))})
        self.check("manifest_file_hashes", not wrong, wrong)

    def zip_matches(self, filename: str, folder: Path):
        path = self.package / filename
        expected = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}
        with zipfile.ZipFile(path) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            unsafe = [item.filename for item in archive.infolist() if not safe_member(item.filename) or stat.S_ISLNK(item.external_attr >> 16)]
            names = [item.filename for item in infos]
            self.check(filename + "_zip_safe", not unsafe and len(names) == len(set(names)), {"unsafe": unsafe})
            # Either direct relative entries or one named container folder is portable.
            if set(names) == set(expected):
                prefix = ""
            else:
                prefixes = {name.split("/", 1)[0] for name in names}
                prefix = next(iter(prefixes)) + "/" if len(prefixes) == 1 else ""
            mapped = {item.filename[len(prefix):]: item for item in infos if item.filename.startswith(prefix)}
            self.check(filename + "_exact_file_set", set(mapped) == set(expected),
                       {"missing": sorted(set(expected) - set(mapped)), "unexpected": sorted(set(mapped) - set(expected)), "prefix": prefix})
            wrong = []
            for name in set(expected) & set(mapped):
                item = mapped[name]
                if item.file_size != expected[name].stat().st_size:
                    wrong.append(name)
                    continue
                value = hashlib.sha256()
                with archive.open(item) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        value.update(chunk)
                if value.hexdigest() != digest(expected[name]):
                    wrong.append(name)
            self.check(filename + "_exact_file_contents", bool(expected) and not wrong, wrong)

    def readme_links(self):
        failures, inspected = [], 0
        for path in self.repository.rglob("README.md"):
            text = path.read_text(encoding="utf-8-sig")
            links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
            for target in links:
                target = target.strip()
                target = target[1:target.index(">")] if target.startswith("<") and ">" in target else target.split(' "', 1)[0]
                parts = urlsplit(target)
                if parts.scheme or target.startswith(("#", "//", "{{", "<")):
                    continue
                relative = unquote(parts.path)
                if not relative:
                    continue
                inspected += 1
                resolved = (path.parent / relative).resolve()
                if not resolved.is_relative_to(self.repository) or not resolved.exists():
                    failures.append({"readme": self.relative(path), "target": target})
        self.check("readme_relative_links_exist_in_repository", not failures, {"inspected": inspected, "broken": failures})

    def run(self):
        tasks = (
            ("required_files", self.required_files),
            ("public_privacy", self.public_privacy),
            ("checkpoint_containers", self.checkpoint_containers),
            ("python_sources", self.python_sources),
            ("notebook", self.notebook),
            ("evaluation_metrics", self.evaluation_metrics),
            ("checksum_file", self.checksum_file),
            ("manifest_hashes", self.manifest_hashes),
            ("source_archive", lambda: self.zip_matches("smart-agriculture-source.zip", self.repository)),
            ("release_archive", lambda: self.zip_matches("smart-agriculture-release.zip", self.assets)),
            ("readme_links", self.readme_links),
        )
        for name, function in tasks:
            self.guarded(name, function)
        failed = [item for item in self.checks if not item["passed"]]
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if not failed else "failed",
            "passed": not failed,
            "checks_total": len(self.checks), "checks_failed": len(failed),
            "checks": self.checks, "details": self.details,
            "scope": "Public package file integrity, privacy, syntax, declared evaluation arithmetic, and archive portability. No Torch import, GPU use, dataset re-evaluation, or independent proof of notebook execution.",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    report = args.report.resolve() if args.report else root / "reports/submission_verification.json"
    result = Verification(root).run()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks_total": result["checks_total"],
                      "checks_failed": result["checks_failed"], "report": str(report)}, indent=2))
    for item in result["checks"]:
        if not item["passed"]:
            print("FAIL " + item["check"] + ": " + json.dumps(item.get("detail"), ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
