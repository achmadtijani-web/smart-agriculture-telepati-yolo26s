"""Resolve the selected checkpoint's input size without importing the GPU runtime."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def valid_size(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 32:
        raise ValueError("--imgsz harus bilangan bulat positif kelipatan 32.")
    return size


def image_size(weights: Path, override: int | None, root: Path = ROOT) -> int:
    if override is not None:
        return valid_size(override)
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    for relative in ('reports/submission_evaluation/metrics.json', 'reports/evaluation/metrics.json'):
        path = root / relative
        if path.is_file():
            metrics = json.loads(path.read_text(encoding='utf-8-sig'))
            if metrics.get('weights_sha256') == weight_hash:
                return valid_size(metrics['imgsz'])
    selection_path = root / "selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
        if selection.get("weights_sha256") == weight_hash:
            return valid_size(selection.get('inference',{}).get('imgsz',selection["winner"]["config"]["imgsz"]))
    return 640


def confidence(weights: Path, override: float | None, root: Path = ROOT) -> float:
    """Use validation's operating threshold only when it belongs to these weights."""
    def valid(value):
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError("Confidence harus lebih besar dari 0 dan maksimal 1.")
        return float(value)

    if override is not None:
        return valid(override)
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    for relative in ('reports/submission_evaluation/metrics.json', 'reports/evaluation/metrics.json'):
        path = root / relative
        if path.is_file():
            metrics = json.loads(path.read_text(encoding='utf-8-sig'))
            if metrics.get('weights_sha256') == weight_hash:
                return valid(metrics['selected_confidence'])
    return 0.25
