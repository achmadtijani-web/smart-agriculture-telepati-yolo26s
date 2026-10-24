"""Inferensi deteksi lokal dengan Ultralytics; setiap hasil ditulis secara bertahap."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
IMAGE_SUFFIXES = {
    ".avif", ".bmp", ".dng", ".heic", ".heif", ".jp2", ".jpeg", ".jpg", ".mpo",
    ".png", ".tif", ".tiff", ".webp",
}
VIDEO_SUFFIXES = {
    ".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".ts", ".webm", ".wmv",
}
CSV_FIELDS = [
    "result_index", "source", "frame_index", "class_id", "class_name",
    "confidence", "x_min", "y_min", "x_max", "y_max",
]
FRAME_CSV_FIELDS = [
    "result_index", "source", "frame_index", "detections_count", "inference_ms",
    "at_detection_limit",
]
NO_DETECTION_NOTE = "Tidak ada deteksi bukan bukti tanaman sehat."
COUNT_NOTE = (
    "Jumlah adalah kemunculan deteksi per gambar/frame, bukan jumlah tanaman unik. "
    "Deteksi tidak membuktikan diagnosis agronomis, tingkat keparahan penyakit, "
    "atau rekomendasi penanganan."
)


def root_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Nilai harus lebih besar dari 0.")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Nilai harus lebih besar dari 0 dan maksimal 1.")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi objek tanaman dari gambar, folder, video, atau kamera lokal.",
        epilog=("Semua jalur relatif dihitung dari folder proyek. " + COUNT_NOTE),
    )
    parser.add_argument("--weights", default="models/best.pt",
                        help="Checkpoint .pt model terlatih yang sudah tersedia secara lokal.")
    parser.add_argument("--source", required=True,
                        help="Gambar, folder media, video lokal, atau indeks kamera (misalnya 0).")
    parser.add_argument("--output", default="outputs/predictions",
                        help="Folder induk untuk run baru; hasil lama tidak ditimpa.")
    parser.add_argument("--conf", type=probability, default=None,
                        help="Bawaan mengikuti ambang validation jika hash bobot cocok; bobot lain 0.25.")
    parser.add_argument("--iou", type=probability, default=0.7,
                        help="Ambang IoU untuk penyaringan deteksi (default: 0.7).")
    parser.add_argument("--imgsz", type=positive_int, default=None,
                        help="Bawaan mengikuti model terpilih; bobot lain memakai 640.")
    parser.add_argument("--max-det", type=positive_int, default=300,
                        help="Batas deteksi per gambar/frame (default: 300); naikkan untuk tanaman rapat.")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, atau indeks GPU seperti 0. auto memakai CUDA jika tersedia.")
    parser.add_argument("--max-frames", type=positive_int,
                        help="Batas total gambar/frame yang diproses, berguna untuk kamera/video.")
    parser.add_argument("--no-save-media", action="store_true",
                        help="Simpan CSV dan JSON saja tanpa gambar/video beranotasi.")
    return parser.parse_args(argv)


def validate_source(value: str) -> tuple[str | int, str]:
    if re.fullmatch(r"[0-9]+", value):
        return int(value), "camera"
    if "://" in value:
        raise ValueError("Gunakan berkas/folder lokal atau indeks kamera; URL tidak didukung.")
    source = root_path(value)
    if not source.exists():
        raise ValueError(f"Sumber tidak ditemukan: {source}")
    if source.is_dir():
        if not any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
                   for p in source.iterdir()):
            raise ValueError("Folder sumber tidak berisi gambar/video yang didukung "
                             "pada tingkat folder tersebut.")
        return str(source), "directory"
    if not source.is_file():
        raise ValueError(f"Sumber harus berupa berkas atau folder: {source}")
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return str(source), "image"
    if source.suffix.lower() in VIDEO_SUFFIXES:
        return str(source), "video"
    raise ValueError(f"Format sumber tidak didukung: {source.suffix}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate_run(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stem = "predict-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    for index in range(10000):
        name = stem if index == 0 else f"{stem}-{index:02d}"
        run = output / name
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise RuntimeError("Tidak dapat membuat folder run unik.")


def resolve_device(requested: str) -> str:
    if requested.strip().lower() == "auto":
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    requested = requested.strip().lower()
    if requested == "cpu" or re.fullmatch(r"[0-9]+(?:,[0-9]+)*", requested):
        return requested
    raise ValueError("--device harus auto, cpu, atau indeks GPU seperti 0 atau 0,1.")


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def close_predictor(model: Any, stream: Any) -> None:
    """Lepaskan kamera/video ketika generator dihentikan pada --max-frames."""
    close = getattr(stream, "close", None)
    if callable(close):
        # Ultralytics 8.4.182 stream_inference() memakai finally untuk melepaskan
        # writer, menutup prefetch worker, lalu menutup sumber video/kamera.
        close()
        return
    # Jika pembuatan generator gagal, predictor mungkin sudah membuka sumber.
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    writers = getattr(predictor, "vid_writer", {})
    if isinstance(writers, dict):
        for writer in writers.values():
            writer.release()
    dataset = getattr(predictor, "dataset", None)
    dataset_close = getattr(dataset, "close", None)
    if callable(dataset_close):
        dataset_close()
    else:
        capture = getattr(dataset, "cap", None)
        if capture is not None:
            capture.release()


def run_prediction(args: argparse.Namespace, model_class: Any = None) -> Path:
    weights = root_path(args.weights)
    if not weights.is_file():
        raise ValueError(f"Bobot model tidak ditemukan: {weights}. Jalankan pelatihan terlebih dahulu.")
    if weights.suffix.lower() != ".pt":
        raise ValueError("--weights harus checkpoint .pt hasil pelatihan, misalnya weights/best.pt.")
    from runtime_config import image_size, confidence
    args.imgsz = image_size(weights, args.imgsz)
    args.conf = confidence(weights, args.conf)
    source, source_type = validate_source(args.source)
    device = resolve_device(args.device)
    if model_class is None:
        from ultralytics import YOLO

        model_class = YOLO
    model = model_class(str(weights))
    if getattr(model, "task", None) != "detect":
        raise ValueError("Bobot harus merupakan model Object Detection (task=detect).")
    output = allocate_run(root_path(args.output))
    counts: Counter[int] = Counter()
    class_names: dict[int, str] = {}
    frames = 0
    stop_reason = "source_exhausted"
    last_source: str | None = None
    frame_index = 0
    stream = None
    started = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started,
        "model": {"path": str(weights), "sha256": sha256_file(weights)},
        "config": {
            "source": source, "source_type": source_type, "device": device,
            "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
            "max_det": args.max_det,
            "nms": None,
            "max_frames": args.max_frames, "save_media": not args.no_save_media,
            "frame_index_base": 0,
            "frame_index_description": "Urutan frame yang diproses per sumber; kamera dapat melewati frame.",
            "coordinate_space": "xyxy dalam piksel gambar/frame asli",
        },
        "output_directory": str(output),
        "frames_csv": str(output / "frames.csv"),
        "frame_interpretation": NO_DETECTION_NOTE + " frames.csv mencatat setiap gambar/frame yang diproses, termasuk tanpa deteksi; at_detection_limit menandai hasil yang mencapai batas max_det.",
        "count_interpretation": COUNT_NOTE,
    }
    try:
        with (output / "detections.csv").open("w", newline="", encoding="utf-8-sig") as handle, \
                (output / "frames.csv").open("w", newline="", encoding="utf-8-sig") as frame_handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            frame_writer = csv.DictWriter(frame_handle, fieldnames=FRAME_CSV_FIELDS)
            frame_writer.writeheader()
            # stream=True menghindari penumpukan Results untuk video atau banyak gambar.
            stream = model.predict(
                source=source, stream=True, save=not args.no_save_media,
                project=str(output.parent), name=output.name, exist_ok=True,
                conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=device,
                max_det=args.max_det, batch=1, nms=None,
                save_txt=False, save_crop=False, show=False, verbose=False,
                stream_buffer=False,
            )
            for result in stream:
                result_source = str(result.path)
                frame_index = frame_index + 1 if result_source == last_source else 0
                last_source = result_source
                names = result.names.items() if isinstance(result.names, dict) else enumerate(result.names)
                class_names.update((int(key), str(name)) for key, name in names)
                boxes = result.boxes
                detections_count = 0
                if boxes is not None:
                    coordinates = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    class_ids = boxes.cls.cpu().tolist()
                    if not (len(coordinates) == len(confidences) == len(class_ids)):
                        raise RuntimeError("Jumlah koordinat, confidence, dan kelas tidak cocok.")
                    detections_count = len(coordinates)
                    for xyxy, confidence, class_id in zip(coordinates, confidences, class_ids):
                        class_id = int(class_id)
                        class_name = str(result.names[class_id])
                        class_names[class_id] = class_name
                        counts[class_id] += 1
                        writer.writerow({
                            "result_index": frames, "source": result_source,
                            "frame_index": frame_index, "class_id": class_id,
                            "class_name": class_name, "confidence": round(float(confidence), 6),
                            **dict(zip(["x_min", "y_min", "x_max", "y_max"],
                                       (round(float(coordinate), 3) for coordinate in xyxy))),
                        })
                inference_ms = (getattr(result, "speed", None) or {}).get("inference")
                frame_writer.writerow({
                    "result_index": frames, "source": result_source,
                    "frame_index": frame_index, "detections_count": detections_count,
                    "inference_ms": round(float(inference_ms), 3) if inference_ms is not None else None,
                    "at_detection_limit": detections_count >= args.max_det,
                })
                frames += 1
                handle.flush()
                frame_handle.flush()
                if args.max_frames is not None and frames >= args.max_frames:
                    stop_reason = "max_frames"
                    break
        if frames == 0:
            raise RuntimeError("Sumber tidak menghasilkan gambar/frame yang dapat dibaca.")
        summary["status"] = "completed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        stop_reason = "keyboard_interrupt"
        raise
    except Exception as exc:
        summary["status"] = "failed"
        stop_reason = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            close_predictor(model, stream)
        except Exception as exc:
            summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if summary["status"] == "completed":
                summary["status"] = "failed"
                stop_reason = "cleanup_error"
                raise
        finally:
            summary.update({
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "frames_processed": frames,
                "stop_reason": stop_reason,
                "detections_total": sum(counts.values()),
                "detections_per_class": [
                    {"class_id": class_id, "class_name": name, "count": counts[class_id]}
                    for class_id, name in sorted(class_names.items())
                ],
            })
            write_json(output / "summary.json", summary)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = run_prediction(args)
    except KeyboardInterrupt:
        print("Inferensi dihentikan. Hasil parsial tersimpan di folder run.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Gagal menjalankan inferensi: {exc}", file=sys.stderr)
        return 1
    print(f"Hasil inferensi: {output}")
    print(f"CSV deteksi: {output / 'detections.csv'}")
    print(f"CSV setiap gambar/frame: {output / 'frames.csv'}")
    print(f"Ringkasan: {output / 'summary.json'}")
    print(COUNT_NOTE)
    print(NO_DETECTION_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
