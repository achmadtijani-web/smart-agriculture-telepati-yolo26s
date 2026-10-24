# Smart Agriculture — deteksi objek tanaman padi

Model yang disubmit: **yolo26s_24b16**, dengan **mAP@50 66.80%** dan
**macro-F1 67.31%** pada 1.059 gambar test internal.
Checkpoint dipilih berdasarkan validation mAP@50–95; test juga pernah dipakai dalam
eksperimen sebelumnya dan belum menjadi pengujian lapangan independen.

Kode ini mengenali bounding box dan 18 label anotasi Telepati. Label serupa tetap
terpisah. Model belum membuktikan diagnosis agronomi, penghitungan tanaman unik,
deteksi seluruh penyakit, atau kinerja kamera drone di sawah.

## Berkas dan hasil

- `Smart_Agriculture_Experiments.ipynb`: notebook yang memuat eksperimen, hasil nyata,
  dan sel reproduksi. Output yang disimpan hanya analisis laporan melalui CPU;
  sel instalasi, pelatihan, evaluasi, dan inferensi dimatikan secara bawaan.
- `reports/evaluation/metrics.json`: metrik bobot final yang tersedia di Release.
- `reports/evaluation/per_class_metrics.csv`: 18 kelas pada setiap split val/test.
- `selection.json`, `experiment_plan.json`, metadata training, dan history:
  bukti pemilihan model melalui validation.
- `MODEL_CARD.md`: protokol dan keterbatasan; `THIRD_PARTY_NOTICES.md`: atribusi.
- `docs/PANDUAN_UPLOAD_GITHUB.md` dan `docs/ISIAN_FORMULIR.md`: cara unggah dan isian.
- `docs/DIAGNOSIS_DATA_MODEL.md`: AP per kelas, label, keseimbangan, ukuran objek,
  matriks operasional, dan uji resolusi; RF-DETR hanya diuji kelayakan perangkat.

## Menggunakan bobot yang disubmit

Unduh `best.pt` dan notebook dari **GitHub Release milik repositori ini**, setelah
rilis diterbitkan. Letakkan bobot di `models/best.pt`; bobot tidak disimpan sebagai
berkas Git biasa. SHA256 final: `2b4a698f807590bea2a8f91bcd9d7b54b52f3b48d76640a421fb9fb9398d959b`.
Unduh juga `yolo11s_parent.pt`, `yolo26n_reference.pt` ke `models/` untuk resep yang memakai checkpoint pendahulu.

Gunakan Python 3.11. Pada Windows dengan GPU NVIDIA yang sesuai:

```powershell
./setup.ps1
.venv/Scripts/python.exe predict.py --weights models/best.pt --source path/gambar.jpg --device auto
```

Untuk Linux/macOS atau CPU, buat virtual environment, instal Torch 2.7.1 dan
Torchvision 0.22.1 dari [indeks resmi yang sesuai](https://pytorch.org/get-started/previous-versions/),
lalu `python -m pip install -r requirements.txt`. Indeks CUDA pada `setup.ps1`
bersifat khusus lingkungan eksperimen; notebook menjelaskan pilihan indeks CPU.
Inferensi mengikuti ukuran input dan confidence di laporan evaluasi bila hash
model cocok. Ukuran pelatihan dapat berbeda dari ukuran inferensi yang dipilih
dari uji resolusi validation. Untuk video, gunakan path video pada `--source`.
Pemeriksaan konfigurasi tanpa GPU: `python -B -m unittest test_runtime_config -v`.

## Menyiapkan dataset dan mengulang eksperimen

Dataset asli **tidak didistribusikan** di repositori atau Release. Sediakan sendiri
`Telepati 8.0 Datasets.zip` dari penyelenggara dengan izin penggunaannya. Ekstrak
ke `data/raw/`; temukan folder yang langsung memuat `train/`, `valid/`, `test/`
dan anotasi COCO. Jangan mengubah nama kelas atau memindahkan data test ke train.

```powershell
python prepare_dataset.py --source "data/raw/Telepati 8.0 Datasets" --output data/yolo --reports reports/reproduction_dataset
python run_recipe.py --list
python run_recipe.py --candidate NAMA_DARI_PLAN --data data/yolo/data.yaml --device 0
```

Perintah preparation membuat tujuan baru dan menulis audit baru. Audit rekaman
tetap berada di `reports/dataset_audit.json`. Siapkan split yang sama sebelum
menafsirkan hasil reproduksi; YAML mengandung root dataset lokal sehingga hash
byte YAML pada mesin baru dapat berbeda. Hash tersebut merekam konfigurasi asli,
bukan hash universal seluruh dataset. Pengelompokan duplikasi menggunakan hash
byte/piksel dan keluarga nama ekspor, dengan prioritas test > val > train.
Jumlah gambar bersih: train 8.822, val 2.046, test 1.059. Tidak ada kelompok
lintas split yang tersisa berdasarkan proksi tersebut; kesamaan sumber/lahan,
sesi pemotretan, dan duplikasi perseptual yang belum terdeteksi tetap mungkin.

Resep fine-tuning YOLO11s memakai `models/yolo11s_parent.pt` dari Release,
yang sudah dilatih 12 epoch pada dataset ini. Itu merupakan warm start baru,
bukan resume optimizer. YOLO26s dimulai dari bobot dasar resmi `yolo26s.pt`
(diunduh Ultralytics bila diperlukan). Metadata mencatat epoch yang benar-benar
selesai serta konfigurasi tiap resep. Arsitektur, jumlah epoch, warm start, dan
resolusi berbeda sehingga perubahan skor tidak mengisolasi satu faktor.

Untuk mengevaluasi bobot final, jalankan setelah data tersedia:

```powershell
python evaluate.py --weights models/best.pt --data data/yolo/data.yaml --output reports/evaluation_reproduced --device 0
```

Evaluasi mengambil confidence dari macro-F1 validation, lalu memakai confidence
yang sama pada test. F1 submission adalah rata-rata F1 **per kelas** pada IoU 0,50,
bukan harmonic mean dari rata-rata precision/recall dan bukan classification
accuracy. mAP@50 dihitung dari kurva AP lintas confidence pada IoU 0,50.
Angka dapat sedikit berubah menurut perangkat/versi; satu seed tidak mengukur
ketidakpastian eksperimen. Periksa hasil per kelas, khususnya `Brown spot`,
sebelum menggunakannya untuk mengambil keputusan.

## Mengunggah

Upload isi folder sumber ke repositori publik. Buat Release dan upload **`best.pt`
serta notebook sebagai aset terpisah**, bersama metrik, model card, dan checksum.
Tiga URL contoh dalam panduan adalah template; tidak ada tautan publik yang dibuat
otomatis oleh paket lokal ini. Ikuti `docs/PANDUAN_UPLOAD_GITHUB.md`.
