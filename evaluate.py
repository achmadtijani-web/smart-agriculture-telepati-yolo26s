"""Evaluate once on validation and held-out test, with validation-only threshold selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi mAP dan ambang confidence dari validation.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/yolo/data.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Bawaan mengikuti model terpilih di selection.json; bobot lain memakai 640.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f"File tidak ditemukan: {path}")
    from runtime_config import image_size
    args.imgsz = image_size(args.weights, args.imgsz)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Folder evaluasi sudah berisi hasil. Gunakan --output baru.")
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    args.output.mkdir(parents=True, exist_ok=True)
    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = YOLO(str(args.weights.resolve()))
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), weights=str(args.weights.resolve()),
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        data_yaml=str(args.data.resolve()),
        data_yaml_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), device=device,
        ultralytics=ultralytics.__version__, torch=torch.__version__, imgsz=args.imgsz,
        nms_iou=0.7, ap_confidence_floor=0.001, max_det=300,
        nms=None, inference_mode_note="Ultralytics default NMS; YOLO26 uses the one-to-many head.",
        confusion_matrix_plot=dict(confidence=0.001, matching_iou=0.45),
        threshold_selection="Confidence dengan macro-F1 validation tertinggi pada kurva Ultralytics.",
        precision_recall_note="P/R interpolasi kurva pada confidence validation yang sama; IoU matching 0.50. Angka bukan accuracy klasifikasi dan confidence bukan probabilitas terkalibrasi.",
    )
    selected = None
    rows = []
    for split in ("val", "test"):
        metrics = model.val(data=str(args.data.resolve()), split=split, imgsz=args.imgsz,
                            batch=args.batch, workers=args.workers, device=device,
                            conf=0.001, iou=0.7, max_det=300, nms=None, plots=True,
                            project=str(args.output.resolve()), name=split, exist_ok=False,
                            verbose=False, save_json=False)
        box = metrics.box
        matrix = np.asarray(metrics.confusion_matrix.matrix)
        matrix_names = [model.names[i] for i in range(len(model.names))] + ['background']
        with (args.output / split / 'confusion_matrix_counts.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['predicted / actual', *matrix_names])
            writer.writerows([name, *map(int, row)] for name, row in zip(matrix_names, matrix))
        if selected is None:
            eligible = np.flatnonzero(np.asarray(box.px) >= 0.001)
            f1_curve = np.asarray(box.f1_curve).mean(axis=0)
            index = int(eligible[np.argmax(f1_curve[eligible])])
            selected = float(box.px[index])
            summary["selected_confidence"] = selected
            summary["validation_threshold_has_true_positives"] = bool(f1_curve[index] > 0)
        class_ids = np.asarray(box.ap_class_index, dtype=int)
        precisions, recalls, f1s = [], [], []
        for i, cid in enumerate(class_ids):
            p = float(np.interp(selected, box.px, box.p_curve[i]))
            r = float(np.interp(selected, box.px, box.r_curve[i]))
            f1 = 2*p*r/(p+r+1e-16)
            precisions.append(p); recalls.append(r); f1s.append(f1)
            rows.append(dict(split=split, class_id=int(cid), class_name=model.names[int(cid)],
                             gt_instances=int(metrics.nt_per_class[cid]),
                             images_with_class=int(metrics.nt_per_image[cid]),
                             confidence=selected, precision_iou50=p, recall_iou50=r,
                             f1_iou50=f1, ap50=float(box.ap50[i]), ap50_95=float(box.ap[i])))
        summary[split] = dict(mAP50=float(box.map50), mAP50_95=float(box.map),
                             macro_precision_iou50=float(np.mean(precisions)),
                             macro_recall_iou50=float(np.mean(recalls)),
                             macro_f1_iou50=float(np.mean(f1s)),
                             gt_instances=int(np.asarray(metrics.nt_per_class).sum()),
                             classes_evaluated=len(class_ids), speed_ms_per_image=metrics.speed)
    with (args.output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
