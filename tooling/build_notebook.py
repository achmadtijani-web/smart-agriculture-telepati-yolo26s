"""Build a portable submission notebook and execute only its CPU report cells.

No dependency installation, dataset preparation, training, evaluation, inference,
network request, or checkpoint loading is performed by this builder.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def source(text: str) -> list[str]:
    return textwrap.dedent(text).strip().splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str, *, record: bool = False) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["recorded-analysis"]} if record else {},
        "outputs": [],
        "source": source(text),
    }


def cells() -> list[dict]:
    return [
        markdown(r"""
        # Smart Agriculture: eksperimen deteksi objek tanaman padi

        Notebook ini menjelaskan penyiapan dataset, eksperimen model, pemilihan checkpoint,
        evaluasi, dan inferensi. Angka yang tersimpan di output berasal dari berkas laporan
        eksperimen nyata yang disertakan dalam repositori. Output analisis dibangun ulang
        dengan Python pada saat notebook dibuat; sel pelatihan tidak ditandai seolah telah
        dijalankan di dalam notebook.

        **Target:** mendeteksi 18 kelas dari anotasi Telepati 8.0. Keluaran merupakan
        bounding box dan label sesuai dataset, bukan diagnosis agronomi yang telah divalidasi
        di sawah. Gambar drone dan kondisi lapangan baru masih membutuhkan pengujian sendiri.

        **Cara pakai lokal:** notebook dapat berada di folder utama repositori atau di
        subfolder `notebook/`; jalankan Jupyter dari folder repositori. Jika mengunduh
        notebook sebagai asset Release, letakkan bersama folder kode hasil clone atau
        ekstraksi source. **Cara pakai Colab:** clone repositori publik yang
        dikumpulkan melalui panel terminal atau sel kode, ubah direktori kerja ke hasil clone,
        lalu buka notebook ini. Unduh `best.pt` dari GitHub Release ke `models/best.pt`.
        Dataset asli tidak disertakan; sediakan ZIP sendiri dan pastikan izin penggunaannya.

        Semua sakelar yang membutuhkan jaringan, dataset, atau komputasi model dimatikan
        secara bawaan. `Run all` menampilkan analisis tersimpan tanpa memulai pelatihan.
        """),
        code(r"""
        from pathlib import Path
        import csv
        import hashlib
        import json
        import os
        import subprocess
        import sys

        def find_root():
            here = Path.cwd().resolve()
            for candidate in (here, *here.parents):
                if (candidate / "train.py").is_file() and (candidate / "reports").is_dir():
                    return candidate
            raise FileNotFoundError("Jalankan Jupyter dari folder repositori atau subfolder notebook/.")

        ROOT = find_root()
        REPORTS = ROOT / "reports"
        DATA = ROOT / "data"
        DATA_YAML = DATA / "yolo" / "data.yaml"
        WEIGHTS = ROOT / "models" / "best.pt"
        ZIP_PATH = DATA / "Telepati 8.0 Datasets.zip"
        RAW_ROOT = DATA / "raw"
        # RAW_SOURCE dicari otomatis setelah ekstraksi ZIP.
        RAW_SOURCE = None
        DEVICE = "auto"   # Bisa diganti menjadi "0" untuk CUDA, atau "cpu".
        RUN_SETUP = False
        RUN_PREPARE = False
        RUN_TRAINING = False
        RUN_EVALUATION = False
        RUN_INFERENCE = False
        # Gunakan direktori baru agar hasil reproduksi tidak menimpa bukti eksperimen.
        REPRO_NAME = "reproduce_selected"
        EVALUATION_OUTPUT = REPORTS / "evaluation_reproduced"
        SOURCE_IMAGE = ROOT / "examples" / "input.jpg"  # Ganti dengan gambar milik Anda.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")

        def read_json(relative, required=False):
            path = ROOT / relative
            if not path.is_file():
                if required:
                    raise FileNotFoundError(relative)
                return None
            return json.loads(path.read_text(encoding="utf-8-sig"))

        def run_script(script, *arguments):
            subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                           cwd=ROOT, check=True)

        print("Konfigurasi portable siap. Semua operasi model dan instalasi dimatikan.")
        """, record=True),
        markdown(r"""
        ## 1. Lingkungan eksperimen

        Eksperimen referensi memakai Python 3.11, Ultralytics 8.4.182, Torch 2.7.1,
        Torchvision 0.22.1, dan GPU NVIDIA RTX 4050 Laptop 6 GB. Versi CUDA pada mesin
        eksperimen ialah 12.8. Kesamaan seed tidak menjamin hasil bit per bit pada perangkat
        dan versi runtime yang berbeda.

        Sel berikut hanya menginstal jika `RUN_SETUP=True`. Pilih wheel Torch sesuai
        sistem operasi dan driver. Indeks `cu128` ditujukan untuk NVIDIA CUDA; untuk CPU
        ubah ke `https://download.pytorch.org/whl/cpu`. Colab mungkin perlu restart runtime
        setelah instalasi. Sel instalasi tidak dijalankan saat notebook ini disiapkan.
        """),
        code(r"""
        TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
        if RUN_SETUP:
            subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.7.1",
                            "torchvision==0.22.1", "--index-url", TORCH_INDEX], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "ultralytics==8.4.182", "Pillow>=11.0,<13", "PyYAML>=6,<7"], check=True)
        else:
            print("Instalasi dilewati. Aktifkan RUN_SETUP hanya jika lingkungan belum tersedia.")
        """),
        markdown(r"""
        ## 2. Dataset dan audit kebocoran

        Arsip berisi pembagian COCO `train/valid/test`. Pipeline memeriksa gambar,
        kategori, koordinat bounding box, dan duplikasi. Kelompok gambar ditautkan dengan
        SHA-256 berkas, SHA-256 piksel terdekode, serta nama sumber Roboflow. Ketika sebuah
        kelompok tersebar antar split, prioritas dipertahankan `test > val > train` dan
        salinan dari split prioritas lebih rendah dikeluarkan. Duplikasi piksel beserta
        label yang identik dalam split juga dikeluarkan. Ini mengurangi kebocoran yang
        terdeteksi, tetapi tidak membuktikan bebas kemiripan visual atau kebocoran sesi/lahan.

        Pertahankan 18 kelas aktif dan relasi ID COCO ke ID YOLO. Nama yang tampak mirip,
        misalnya `BrownSpot` dan `Brown spot`, tidak otomatis digabung. Dataset memiliki
        perbedaan granularitas anotasi (lesi vs seluruh daun) yang harus diaudit dengan ahli.
        Tiga kategori induk tanpa bounding box dikeluarkan dari pemetaan aktif.
        """),
        code(r"""
        audit = read_json("reports/dataset_audit.json", required=True)
        mapping = read_json("reports/class_mapping.json", required=True)
        print("Status audit:", audit["status"])
        print(f"{'Split':<8} {'Gambar awal':>12} {'Gambar bersih':>14} {'Baris box bersih':>18}")
        for split in ("train", "val", "test"):
            before = audit["original"][split]
            after = audit["prepared"][split]
            print(f"{split:<8} {before['images']:>12,} {after['images']:>14,} {after['annotations']:>18,}")
        print("Jumlah kelas aktif:", mapping["active_class_count"])
        print("Catatan: Ultralytics membuang baris label yang identik saat memuat data;")
        print("jumlah box efektif evaluasi mengikuti metrics.json, bukan sekadar baris TXT.")
        """, record=True),
        code(r"""
        print(f"{'YOLO ID':>7} {'COCO ID':>7}  Nama kategori asli")
        for item in mapping["classes"]:
            print(f"{item['yolo_id']:>7} {item['coco_id']:>7}  {item['name']}")
        """, record=True),
        markdown(r"""
        ## 3. Reproduksi ekstraksi dan penyiapan data

        Simpan ZIP sebagai `data/Telepati 8.0 Datasets.zip`. Sel ini mengekstrak hanya
        ke folder lokal dan menolak entri yang keluar dari folder tujuan. Folder ekstraksi
        harus baru atau kosong. Jalankan sekali dengan `RUN_PREPARE=True`; kemudian matikan
        lagi. `prepare_dataset.py` menerima `--source`, `--output`, `--reports`, dan
        `--workers`. Audit reproduksi disimpan terpisah di `reports/dataset_reproduced`.
        Data hasil reproduksi disimpan di `data/yolo`; pilih lokasi baru jika folder sudah
        berisi data. Pengelompokan yang sama memerlukan arsip asli yang sama.
        """),
        code(r"""
        if RUN_PREPARE:
            import zipfile
            from pathlib import PurePosixPath
            if not ZIP_PATH.is_file():
                raise FileNotFoundError("Sediakan ZIP dataset pada ZIP_PATH.")
            if RAW_ROOT.exists() and any(RAW_ROOT.iterdir()):
                raise FileExistsError("RAW_ROOT harus baru atau kosong; gunakan folder baru.")
            if DATA_YAML.parent.exists() and any(DATA_YAML.parent.iterdir()):
                raise FileExistsError("Folder output dataset harus baru atau kosong.")
            RAW_ROOT.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH) as archive:
                for member in archive.infolist():
                    relative = PurePosixPath(member.filename)
                    if not relative.parts or member.is_dir():
                        continue
                    if relative.parts[0] == "__MACOSX" or relative.name.startswith("."):
                        continue
                    if "\\" in member.filename:
                        raise ValueError("Pemisah path ZIP yang tidak didukung.")
                    destination = (RAW_ROOT / member.filename).resolve()
                    if not destination.is_relative_to(RAW_ROOT.resolve()):
                        raise ValueError("Entri ZIP di luar RAW_ROOT ditolak.")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source_file, destination.open("xb") as target_file:
                        import shutil
                        shutil.copyfileobj(source_file, target_file)
            candidates = [p.parent for p in RAW_ROOT.rglob("train")
                          if p.is_dir() and (p.parent / "valid").is_dir()
                          and (p.parent / "test").is_dir()]
            if len(candidates) != 1:
                raise ValueError("Tidak ditemukan satu folder COCO train/valid/test; tetapkan RAW_SOURCE secara manual.")
            RAW_SOURCE = candidates[0]
            run_script("prepare_dataset.py", "--source", RAW_SOURCE, "--output", DATA_YAML.parent,
                       "--reports", REPORTS / "dataset_reproduced", "--workers", 4)
        else:
            print("Penyiapan dataset dilewati; output audit tersimpan tetap dapat dibaca.")
        """),
        markdown(r"""
        ## 4. Rancangan eksperimen dan pemilihan model

        Baseline awal adalah YOLO11n. Eksperimen berikutnya membandingkan YOLO11s dan
        YOLO26n, masing-masing 12 epoch. Eksperimen lanjutan disimpan dalam folder baru.
        Model lebih besar atau pelatihan lebih lama merupakan kandidat, bukan jaminan
        peningkatan. Pembandingan recipe, resolusi, dan lama pelatihan harus dibaca bersama
        skor; perubahan beberapa faktor tidak mengisolasi pengaruh arsitektur saja.

        **Aturan seleksi:** pilih checkpoint dengan validation mAP@50–95 tertinggi.
        Test tidak digunakan sebagai pemilih epoch, model, confidence, atau augmentasi.
        Split test internal sudah dilihat pada eksperimen sebelumnya; hasilnya adalah
        benchmark internal yang digunakan ulang, belum merupakan validasi eksternal baru.
        """),
        code(r"""
        # Konfigurasi yang benar-benar tercatat; hanya kolom portable ditampilkan.
        wanted = ("model", "epochs", "imgsz", "batch", "optimizer", "lr0", "lrf",
                  "weight_decay", "cos_lr", "warmup_epochs", "warmup_bias_lr", "seed",
                  "deterministic", "mosaic", "close_mosaic", "amp", "nms")
        metadata_paths = sorted(REPORTS.glob("reference_training_*.json"))
        metadata_paths += sorted(REPORTS.glob("training_*.json"))
        for path in metadata_paths:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            config = document.get("config", {})
            print(path.name)
            shown = {key: config[key] for key in wanted if key in config}
            # Sebuah model fine-tune bisa dicatat sebagai path lokal: tampilkan nama berkas saja.
            if isinstance(shown.get("model"), str):
                shown["model"] = shown["model"].replace("\\", "/").rsplit("/", 1)[-1]
            print(json.dumps(shown, ensure_ascii=False, indent=2))
            print("Selesai:", bool(document.get("completed_utc")))
        if not metadata_paths:
            print("Metadata pelatihan belum tersedia.")
        """, record=True),
        code(r"""
        plan = read_json("experiment_plan.json")
        if plan and isinstance(plan.get("candidates"), list):
            print("Kandidat dalam rencana:")
            for candidate in plan["candidates"]:
                shown = {key: candidate[key] for key in ("name", "model", "epochs", "imgsz", "batch", "lr0")
                         if key in candidate}
                if isinstance(shown.get("model"), str):
                    shown["model"] = shown["model"].replace("\\", "/").rsplit("/", 1)[-1]
                print(json.dumps(shown, ensure_ascii=False))

        selection = read_json("selection.json")
        if selection:
            winner = selection.get("winner", {})
            print("Model terpilih:", winner.get("name", "lihat selection.json"))
            print("Epoch terbaik:", winner.get("best_epoch", "lihat metadata"))
            value = winner.get("best_val_map50_95")
            if value is not None:
                print(f"Validation mAP@50–95 saat seleksi: {100 * float(value):.2f}%")
        else:
            print("Seleksi lanjutan belum selesai; skor baru tidak diisi dengan angka perkiraan.")
        """, record=True),
        markdown(r"""
        Sel pelatihan berikut mereproduksi kandidat dengan `train.py` dan konfigurasi
        tercatat. Pilih nama kandidat pada `experiment_plan.json`. Nilai default hanya
        contoh YOLO26n 12 epoch dan diganti otomatis jika nama kandidat ditemukan pada
        rencana. Unduhan pretrained Ultralytics terjadi ketika bobot awal belum tersedia.
        `train.py` merekam konfigurasi lengkap, hash YAML, versi library, dan checkpoint.

        `--model` dapat menunjuk pretrained resmi atau checkpoint fine-tune. Jika rencana
        mencatat path mesin lama, tetapkan `INITIAL_MODEL` ke berkas yang sudah diunduh.
        `--resume` hanya untuk `last.pt` yang belum selesai dan masih menyimpan optimizer.
        Bobot `best.pt` pada Release adalah bobot inferensi, bukan jaminan dapat resume.
        """),
        code(r"""
        # Pilih kandidat pada rencana lanjutan; ganti nama ini untuk kandidat lainnya.
        CANDIDATE_NAME = plan["candidates"][0]["name"] if plan and plan.get("candidates") else "yolo26n_12"
        recipe = {"name": CANDIDATE_NAME, "model": "yolo26n.pt", "epochs": 12,
                  "imgsz": 640, "batch": 16, "lr0": 0.001, "patience": 12}
        if plan:
            matches = [entry for entry in plan.get("candidates", [])
                       if entry.get("name") == CANDIDATE_NAME]
            if matches:
                recipe.update(matches[0])
        INITIAL_MODEL = recipe["model"]  # Ganti jika checkpoint awal merupakan path mesin lain.
        if RUN_TRAINING:
            if not DATA_YAML.is_file():
                raise FileNotFoundError("Siapkan dataset terlebih dahulu.")
            run_script("train.py", "--data", DATA_YAML, "--model", INITIAL_MODEL,
                       "--epochs", recipe["epochs"], "--imgsz", recipe["imgsz"],
                       "--batch", recipe["batch"], "--lr0", recipe["lr0"],
                       "--device", DEVICE, "--workers", 4, "--patience", recipe["patience"],
                       "--seed", 42, "--name", REPRO_NAME)
        else:
            print("Pelatihan dilewati. Aktifkan RUN_TRAINING secara sengaja untuk mereproduksi.")
        """),
        markdown(r"""
        ## 5. Definisi metrik dan ambang confidence

        - **mAP@50:** rata-rata Average Precision 18 kelas pada IoU pencocokan 0,50,
          menggunakan kurva precision-recall di berbagai confidence. Ini bukan accuracy
          klasifikasi dan tidak dihitung hanya pada satu ambang confidence.
        - **mAP@50–95:** rata-rata AP pada IoU 0,50 hingga 0,95 dengan langkah 0,05.
          Metrik ini dipakai untuk pemilihan model melalui validation.
        - **F1 per kelas:** `2 * precision * recall / (precision + recall)` pada IoU 0,50.
          **Macro-F1** ialah rata-rata F1 kelas, bukan F1 dari rata-rata precision dan recall.
          Setiap kelas memiliki bobot sama.
        - **Ambang confidence:** dipilih sekali dari titik macro-F1 maksimum pada kurva
          validation Ultralytics; threshold tersebut digunakan juga untuk test. P/R
          diinterpolasi dari kurva yang sama. NMS IoU 0,70, `max_det=300`, confidence
          floor untuk AP 0,001. Pada YOLO26, `nms=None` memakai jalur default one-to-many
          dengan NMS di versi library ini.

        Skor untuk formulir dikalikan 100. Gunakan mAP@50 dan macro-F1 **test** dari
        checkpoint final yang sama, serta lampirkan definisi macro-F1 di deskripsi Release.
        """),
        code(r"""
        references = [
            ("YOLO11n baseline", "reports/baseline_reference.json"),
            ("YOLO26n referensi", "reports/yolo26n_reference.json"),
        ]
        # Local submission verification can keep its evaluation in a separate folder.
        # The published repository normally stores this as reports/evaluation.
        FINAL_EVALUATION = "reports/submission_evaluation" if (REPORTS / "submission_evaluation" / "metrics.json").is_file() else "reports/evaluation"
        final_metrics = read_json(FINAL_EVALUATION + "/metrics.json")
        if final_metrics:
            final_name = (selection or {}).get("winner", {}).get("name", "model final")
            references.append((str(final_name), FINAL_EVALUATION + "/metrics.json"))
        print(f"{'Model':<29} {'Split':<5} {'mAP50 %':>9} {'mAP50–95 %':>12} {'Macro-F1 %':>11}")
        for name, relative in references:
            document = read_json(relative)
            if not document:
                continue
            for split in ("val", "test"):
                result = document[split]
                print(f"{name:<29} {split:<5} {100*result['mAP50']:>9.2f} "
                      f"{100*result['mAP50_95']:>12.2f} {100*result['macro_f1_iou50']:>11.2f}")
        if final_metrics:
            print("\nNilai formulir untuk checkpoint final:")
            print(f"mAP@50 (%): {100*final_metrics['test']['mAP50']:.2f}")
            print(f"F1-Score macro IoU50 (%): {100*final_metrics['test']['macro_f1_iou50']:.2f}")
            print(f"Confidence dipilih pada validation: {final_metrics['selected_confidence']:.6f}")
            print("Ground truth test efektif:", final_metrics['test']['gt_instances'])
            print("SHA-256 bobot:", final_metrics['weights_sha256'])
        else:
            print("\nEvaluasi final lanjutan belum tersedia; angka referensi bukan klaim hasil model baru.")
        """, record=True),
        code(r"""
        # Evaluasi penuh hanya setelah checkpoint dipilih melalui validation.
        if RUN_EVALUATION:
            if not WEIGHTS.is_file():
                raise FileNotFoundError("Unduh best.pt dari Release ke models/best.pt.")
            if not DATA_YAML.is_file():
                raise FileNotFoundError("Dataset diperlukan untuk evaluasi ulang.")
            run_script("evaluate.py", "--weights", WEIGHTS, "--data", DATA_YAML,
                       "--output", EVALUATION_OUTPUT, "--batch", 8, "--workers", 2,
                       "--device", DEVICE)
        else:
            print("Evaluasi model dilewati. Tabel di atas membaca laporan eksperimen tersimpan.")
        """),
        markdown(r"""
        ## 6. Pemeriksaan per kelas dan keterbatasan

        Lihat kelas dengan AP rendah sebelum memakai model untuk keputusan lapangan.
        Ketidakseimbangan jumlah objek, perbedaan sumber foto, bounding box sangat kecil,
        dan anotasi dengan nama/luas objek berbeda dapat membatasi hasil.
        Peningkatan rata-rata tidak berarti semua kelas meningkat. Kecepatan inferensi
        internal GPU bukan waktu keseluruhan kamera, drone, transmisi, dan pemrosesan.
        """),
        code(r"""
        per_class_path = ROOT / FINAL_EVALUATION / "per_class_metrics.csv"
        if per_class_path.is_file():
            with per_class_path.open(encoding="utf-8-sig", newline="") as stream:
                per_class = [row for row in csv.DictReader(stream) if row["split"] == "test"]
            per_class.sort(key=lambda row: float(row["ap50_95"]))
            print("Lima kelas test dengan mAP@50–95 terendah:")
            for row in per_class[:5]:
                print(f"{row['class_name']:<26} AP50–95={100*float(row['ap50_95']):6.2f}% "
                      f"F1={100*float(row['f1_iou50']):6.2f}%  GT={row['gt_instances']}")
            macro_f1 = sum(float(row["f1_iou50"]) for row in per_class) / len(per_class)
            if final_metrics:
                assert abs(macro_f1 - final_metrics["test"]["macro_f1_iou50"]) < 1e-10
            print("Jumlah kelas test:", len(per_class), "| Macro-F1 cocok dengan metrics.json.")
        else:
            print("Evaluasi per kelas final belum tersedia.")
        """, record=True),
        markdown(r"""
        ## 7. Audit kelas, label, resolusi, dan pembanding

        Diagnosis memakai statistik train dan hasil validation. Label dan 18 kelas tidak
        diubah. Periksa AP per kelas, confusion matrix pada threshold operasional, serta
        perbedaan anotasi lesi dan seluruh daun sebelum menambah model yang lebih besar.
        Berkas lengkap tersedia di `docs/DIAGNOSIS_DATA_MODEL.md` dan `reports/diagnostics/`.
        Uji resolusi berikut menggunakan bobot YOLO26s yang sama pada validation;
        membesarkan gambar sumber 640 tidak otomatis menambah detail. RF-DETR hanya diuji
        untuk kelayakan perangkat, sehingga tidak memiliki skor pembanding final.
        """),
        code(r"""
        diagnosis = read_json("reports/diagnostics/diagnosis.json")
        if diagnosis:
            print("Lima kelas dengan AP validation terendah:")
            for row in sorted(diagnosis["classes"], key=lambda r: r["val_AP50_95_percent"])[:5]:
                print(f"{row['class_name']:<26} AP50={row['val_AP50_percent']:.2f}% "
                      f"AP50–95={row['val_AP50_95_percent']:.2f}% "
                      f"train={row['train_images']} gambar, kecil={row['small32_percent']:.2f}%")
        operating = read_json("reports/diagnostics_operating_val/summary.json")
        if operating:
            print(f"\nConfusion matrix: confidence={operating['confidence']:.6f}, "
                  f"IoU matching={operating['matching_iou']:.2f}; tidak menggantikan F1 kurva.")
            for row in operating["top_class_confusions"][:5]:
                print(f"{row['actual_class']} → {row['predicted_class']}: {row['count']} kotak")
        print("\nInput  Val mAP50%  Val mAP50–95%  Val macro-F1%")
        for size in (640, 768, 960):
            result = read_json(f"reports/resolution_ablation/{size}.json")
            if result:
                print(f"{size:<6} {100*result['mAP50']:>10.2f} "
                      f"{100*result['mAP50_95']:>14.2f} {100*result['macro_f1']:>14.2f}")
        scale_selection = read_json("reports/resolution_ablation/selection.json")
        if scale_selection:
            print("Input terpilih dari validation:", scale_selection["winner"]["imgsz"])
        print("RF-DETR: pilot perangkat saja; pengguna memilih finalisasi YOLO26s untuk submission.")
        """, record=True),
        markdown(r"""
        ## 8. Inferensi pada gambar sendiri

        Simpan satu foto sebagai `examples/input.jpg`, atau ubah `SOURCE_IMAGE` ke gambar
        maupun folder foto lokal. Aktifkan `RUN_INFERENCE=True` setelah mengunduh bobot.
        Threshold mengikuti validation dari model final bila tersedia. Jika belum ada
        metrics.json, 0,25 hanyalah threshold manual untuk demonstrasi. `predict.py`
        mendukung gambar, folder gambar, dan video, lalu menyimpan media beranotasi,
        CSV deteksi, CSV frame termasuk frame tanpa deteksi, serta ringkasan.
        Jumlah deteksi/frame bukan jumlah unik tanaman di seluruh lahan atau video.
        """),
        code(r"""
        if RUN_INFERENCE:
            if not SOURCE_IMAGE.exists():
                raise FileNotFoundError("Tetapkan SOURCE_IMAGE ke gambar atau folder foto lokal.")
            if not WEIGHTS.is_file():
                raise FileNotFoundError("Unduh bobot dari GitHub Release terlebih dahulu.")
            confidence = final_metrics["selected_confidence"] if final_metrics else 0.25
            run_script("predict.py", "--weights", WEIGHTS, "--source", SOURCE_IMAGE,
                       "--output", ROOT / "outputs" / "notebook_predictions",
                       "--conf", confidence, "--device", DEVICE)
        else:
            print("Inferensi dilewati. Isi SOURCE_IMAGE dan aktifkan RUN_INFERENCE untuk mencoba foto.")
        """),
        markdown(r"""
        ## 9. Berkas dan tautan pengumpulan

        1. Buat repositori GitHub **public**, lalu upload kode sumber, `requirements.txt`,
           README, laporan ringkas, dan notebook ini di tingkat utama repositori. Jangan
           upload `.venv`, cache, dataset, token, atau seluruh folder training.
        2. Pada repositori, pilih **Releases → Draft a new release**, buat tag misalnya
           `v1.0.0`, dan upload bobot `best.pt` sebagai asset. Upload juga notebook ini
           sebagai asset bila formulir meminta tautan GitHub Release notebook.
        3. Publish Release, lalu salin URL halaman Release untuk kolom **Link Model AI**.
           Untuk **Link Jupyter Notebook**, salin URL berkas `.ipynb` di repositori atau
           URL Release yang memuat asset notebook. Untuk **Link Repositori**, salin URL
           halaman utama repositori publik.
        4. Pastikan Release, notebook, dan repositori dapat dibuka tanpa login. Masukkan
           dua angka test final yang dicetak pada bagian evaluasi, dalam skala 0–100.

        Tautan berbentuk `https://github.com/<akun>/<repo>/releases/tag/<tag>` baru valid
        setelah publikasi; alamat contoh bukan tautan yang siap dikumpulkan. Cantumkan
        arsitektur, hash bobot, versi library, 18 kelas, confidence validation, definisi
        macro-F1, dan keterbatasan benchmark internal di README/deskripsi Release.

        Rujukan implementasi: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/),
        [konfigurasi Ultralytics](https://docs.ultralytics.com/usage/cfg/), dan
        [panduan GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
        """),
    ]


def validate_run_all(notebook: dict, root: Path) -> int:
    """Execute every code cell with its safe defaults and prohibit subprocess calls."""
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Safe-default notebook attempted a subprocess operation")

    previous_cwd = Path.cwd()
    previous_run = subprocess.run
    prior_environment = {key: os.environ.get(key) for key in ("YOLO_CONFIG_DIR", "YOLO_AUTOINSTALL")}
    before_modules = set(sys.modules)
    namespace = {"__name__": "__notebook_safe_default_validation__"}
    count = 0
    os.chdir(root)
    subprocess.run = forbidden_call
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
                    count += 1
        for flag in ("RUN_SETUP", "RUN_PREPARE", "RUN_TRAINING", "RUN_EVALUATION", "RUN_INFERENCE"):
            assert namespace[flag] is False, flag
        added_modules = set(sys.modules) - before_modules
        assert not any(name.split(".")[0] in ("torch", "ultralytics") for name in added_modules)
    finally:
        subprocess.run = previous_run
        os.chdir(previous_cwd)
        for key, previous in prior_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    return count


def build(root: Path, output: Path) -> dict:
    notebook = {
        "cells": cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9", "file_extension": ".py",
                              "mimetype": "text/x-python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": []},
            "smart_agriculture": {
                "recorded_outputs": "CPU analysis of included experiment artifacts; no training execution claimed",
                "selection_metric": "validation mAP50-95",
                "default_heavy_operations": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Stable IDs meet notebook 4.5 schema without requiring nbformat to build.
    namespace = {"__name__": "__notebook_analysis__"}
    count = 0
    prior_cwd = Path.cwd()
    prior_yolo_config = os.environ.get("YOLO_CONFIG_DIR")
    prior_yolo_autoinstall = os.environ.get("YOLO_AUTOINSTALL")
    os.chdir(root)
    try:
        for number, cell in enumerate(notebook["cells"], 1):
            cell["id"] = f"agriculture-{number:03d}"
            if cell["cell_type"] != "code":
                continue
            content = "".join(cell["source"])
            ast.parse(content, filename=f"notebook-cell-{number}")
            if "recorded-analysis" in cell["metadata"].get("tags", []):
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    exec(compile(content, f"notebook-cell-{number}", "exec"), namespace)
                count += 1
                cell["execution_count"] = count
                cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                                    "text": captured.getvalue().splitlines(keepends=True)}]
    finally:
        os.chdir(prior_cwd)
        for key, previous in (("YOLO_CONFIG_DIR", prior_yolo_config),
                              ("YOLO_AUTOINSTALL", prior_yolo_autoinstall)):
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    # The public notebook must not include personal paths or user names.
    forbidden = (str(root), str(root).replace("\\", "/"),
                 "C:" + "\\" + "Users" + "\\", "C:" + "/" + "Users" + "/", "AppData")
    def string_values(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for child in value:
                yield from string_values(child)
        elif isinstance(value, dict):
            for child in value.values():
                yield from string_values(child)
    if any(term and term.casefold() in value.casefold()
           for value in string_values(notebook) for term in forbidden):
        raise ValueError("Notebook contains a machine-specific path.")
    safe_default_cells = validate_run_all(notebook, root)
    # Record exact portable evidence files consumed by the CPU analysis. File hashes
    # prove which saved experiments were summarized without claiming GPU execution.
    evidence_paths = [
        root / "reports" / "baseline_reference.json",
        root / "reports" / "yolo26n_reference.json",
        root / "reports" / "dataset_audit.json",
        root / "reports" / "class_mapping.json",
        root / "experiment_plan.json",
        root / "selection.json",
        root / "reports" / "diagnostics" / "diagnosis.json",
        root / "reports" / "diagnostics_operating_val" / "summary.json",
        root / "reports" / "resolution_ablation" / "selection.json",
    ]
    evidence_paths += [root / "reports" / "resolution_ablation" / f"{size}.json"
                       for size in (640, 768, 960)]
    evidence_paths += sorted((root / "reports").glob("reference_training_*.json"))
    evidence_paths += sorted((root / "reports").glob("training_*.json"))
    final_evaluation = root / "reports" / "submission_evaluation"
    if not (final_evaluation / "metrics.json").is_file():
        final_evaluation = root / "reports" / "evaluation"
    evidence_paths += [final_evaluation / "metrics.json", final_evaluation / "per_class_metrics.csv"]
    notebook["metadata"]["execution_provenance"] = {
        "description": (
            "CPU-only analysis cells executed by tooling/build_notebook.py with Python. "
            "Recorded outputs summarize included experiment reports, not GPU training in this notebook. "
            "Safe-default Run all also passed with subprocess calls prohibited, no Torch or Ultralytics "
            "imports, and setup/preparation/training/evaluation/inference flags all False."
        ),
        "source": "tooling/build_notebook.py",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recorded_output_cell_ids": [cell["id"] for cell in notebook["cells"]
                                     if cell.get("outputs")],
        "recorded_cpu_analysis_cells": count,
        "safe_default_code_cells_executed": safe_default_cells,
        "model_operations_executed": False,
        "input_artifacts": [
            {"path": path.relative_to(root).as_posix(),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in evidence_paths if path.is_file()
        ],
    }
    serialized = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return {"output": output.relative_to(root).as_posix(), "cells": len(notebook["cells"]),
            "executed_cpu_cells": count, "code_cells_ast_valid": True,
            "safe_default_code_cells_executed": safe_default_cells,
            "model_operations_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "notebook" / "smart_agriculture_experiments.ipynb").resolve()
    print(json.dumps(build(root, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
