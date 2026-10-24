"""Train a reproducible local YOLO detection baseline; no test-set tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih detektor padi dengan validasi terpisah.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/yolo/data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="experiment")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.workers < 0:
        parser.error("epochs/batch harus positif, imgsz >=32, workers >=0")
    if not args.resume and not args.data.is_file():
        parser.error(f"Dataset YAML tidak ditemukan: {args.data}")
    if args.resume and not args.resume.is_file():
        parser.error(f"Checkpoint resume tidak ditemukan: {args.resume}")
    import torch
    import ultralytics
    from ultralytics import YOLO

    # The application owns local checkpoints and downloads the named official pretrained model.
    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if not args.resume and (ROOT / "runs" / args.name).exists():
        parser.error("Nama run sudah ada. Gunakan --name baru atau --resume path/last.pt.")
    model = YOLO(str(args.resume.resolve()) if args.resume else args.model)
    if args.resume and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
        parser.error("Checkpoint sudah selesai/stripped dan tidak bisa di-resume. Gunakan --model checkpoint.pt --name run_baru untuk fine-tuning baru.")
    config = dict(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=device, patience=args.patience,
        seed=args.seed, deterministic=True, project=str(ROOT / "runs"), name=args.name,
        exist_ok=False, pretrained=True, optimizer="AdamW", lr0=args.lr0,
        lrf=0.01, weight_decay=0.0005, cos_lr=True, warmup_epochs=2, warmup_bias_lr=0.0,
        amp=device != "cpu", cache=False, plots=True, save=True,
        mosaic=0.5, close_mosaic=min(5, args.epochs), mixup=0.0,
        degrees=10.0, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        translate=0.1, scale=0.3, val=True, verbose=False,
    )
    if args.resume:
        # Resume uses saved training options; do not mislabel CLI defaults as executed options.
        config = dict(model.ckpt.get("train_args", {}))
        config.update(device=device, workers=args.workers, resume=str(args.resume.resolve()))
    actual_data = Path(config["data"])
    metadata = dict(
        started_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        torch=torch.__version__, ultralytics=ultralytics.__version__,
        cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        config=config, data_yaml_sha256=hashlib.sha256(actual_data.read_bytes()).hexdigest(),
        note="Best checkpoint dipilih hanya dengan validation mAP50-95; test tidak dipakai melatih.",
    )
    (ROOT / "reports").mkdir(exist_ok=True)
    metadata_name = args.resume.resolve().parent.parent.name if args.resume else Path(args.name).name
    metadata_path = ROOT / "reports" / f"training_{metadata_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.resume:
        model.train(resume=True, device=device, workers=args.workers)
    else:
        model.train(**config)
    run = Path(model.trainer.save_dir)
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
                    config=vars(model.trainer.args),
                    best_checkpoint=str(run / "weights/best.pt"))
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Selesai. Bobot terbaik: {run / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
