# Model Card — yolo26s_24b16

## Identitas dan tujuan

- Tugas: object detection pada gambar tanaman/daun padi, dengan 18 kelas asli aktif.
- Bobot publik: `best.pt`; SHA256 `2b4a698f807590bea2a8f91bcd9d7b54b52f3b48d76640a421fb9fb9398d959b`.
- Hash checkpoint sumber sebelum serialisasi metadata publik:
  `232fd2295d08cf482a9ec99092a757caa0bff350d65593e33d2beff5645f74b0`.
- Sanitasi mengganti metadata path lokal; laporan publikasi memeriksa bahwa semua
  tensor model tetap sama. Metrik final dihitung ulang dari **bobot publik**.
- Kegunaan: eksperimen pemantauan visual dan peninjauan gejala oleh manusia.
- Belum divalidasi: diagnosis di sawah, kendali pestisida otomatis, hitungan
  tanaman unik, pemetaan drone, video lintas musim/lahan, atau penyakit di luar kelas.

## Data dan protokol

Dataset input: ZIP Telepati 8.0 yang disediakan pengguna/penyelenggara; tidak
disertakan dalam paket. Kelas tidak digabung meskipun namanya mirip karena
granularitas anotasi berbeda (gejala kecil versus seluruh daun).
Tiga kategori induk tanpa bounding box dikeluarkan. Data bersih terdiri dari
8.822 gambar train, 2.046 val, dan 1.059 test. Efektif loader:
18.002 kotak train, 4.758 val, 2.669 test; beberapa baris anotasi identik
dihapus loader Ultralytics. Audit mempertahankan rincian jumlah baris asal.

Duplikasi ditangani dengan SHA256 berkas, piksel RGB, dan keluarga nama ekspor.
Prioritas test > val > train; kelompok yang sama dikeluarkan dari split lebih
rendah. Proksi ini tidak menyingkirkan seluruh kemungkinan kesamaan sumber,
lahan, sesi, atau duplikasi perseptual. Test internal telah dipakai pada
eksperimen sebelumnya; skor ini belum merupakan benchmark eksternal tersembunyi.

Checkpoint dipilih melalui **validation mAP@50–95**, tanpa memilih berdasarkan
test. Resep menguji kapasitas/epoch/resolusi/warm start berbeda dengan seed 42;
hasil bukan isolasi pengaruh arsitektur dan belum memuat rerata beberapa seed.
YOLO11s pendahulu dilatih 12 epoch; fine-tuning, bila terpilih, menambah epoch
baru dari checkpoint tersebut. `selection.json` dan history mencatat durasi
aktual, bukan mengasumsikan semua run mencapai batas epoch.

Ukuran input evaluasi final: `640`. AP confidence floor:
`0.001`; NMS IoU `0.7`;
max detections `300`. Ultralytics default NMS digunakan;
untuk YOLO26 ini adalah head one-to-many. Confidence F1 dipilih dari validation:
`0.26626626626626626`. P/R diinterpolasi dari kurva pada IoU 0,50.
F1 adalah rata-rata harmonic mean P/R setiap kelas. Confidence tidak dibuktikan
sebagai probabilitas diagnosis yang terkalibrasi.

## Hasil test internal

| Metrik | Persentase |
|---|---:|
| mAP@50 | 66.80 |
| mAP@50–95 | 47.25 |
| Macro precision, IoU 0,50 | 69.01 |
| Macro recall, IoU 0,50 | 71.12 |
| Macro F1, IoU 0,50 | 67.31 |

Lima kelas dengan AP@50–95 terendah pada test:

| Kelas | AP@50 (%) | AP@50–95 (%) | F1 (%) | Kotak GT |
|---|---:|---:|---:|---:|
| Brown spot | 11.97 | 4.77 | 17.39 | 465 |
| Bacterial leaf blight | 34.25 | 18.63 | 39.51 | 59 |
| Leaf Scald | 39.03 | 22.45 | 44.30 | 187 |
| BrownSpot | 50.81 | 22.86 | 55.04 | 375 |
| Blast | 45.02 | 23.95 | 46.08 | 449 |

Skor keseluruhan dapat menyembunyikan kelas lemah dan perbedaan prevalensi.
Salah deteksi serta objek terlewat harus diperiksa manusia. Gunakan foto dari
perangkat, lahan, musim, pencahayaan, dan ketinggian drone sasaran untuk validasi
lanjutan sebelum keputusan agronomi.

## Reproduksi dan dependensi

Lihat README dan notebook. Lingkungan rekaman: Python 3.11, Torch 2.7.1+cu128,
Ultralytics 8.4.182, NVIDIA RTX 4050 Laptop 6 GB. Bobot dasar YOLO bersumber dari
Ultralytics. Aset pendahulu/referensi publik: yolo11s_parent.pt, yolo26n_reference.pt.
Riwayat evaluasi referensi tetap menyimpan hash file asli; serialisasi publik
referensi memiliki hash berbeda yang dicatat terpisah, dengan tensor sama.
Paket tidak menyatakan lisensi dataset baru atau URL lisensi yang tidak tersedia;
perhatikan `THIRD_PARTY_NOTICES.md` dan ketentuan penyelenggara.

## Daftar kelas

| ID | Nama persis anotasi |
|---:|---|
| 0 | Bacterial panicle Blight |
| 1 | False-Smut |
| 2 | Healthy Rice Leaf |
| 3 | Healthy Rice beads |
| 4 | Infected Blast |
| 5 | Leaf-roller |
| 6 | Blast |
| 7 | BrownSpot |
| 8 | Healthy |
| 9 | Leaf Scald |
| 10 | Rice-Tungro |
| 11 | Sheath Blight |
| 12 | Bacterial leaf blight |
| 13 | Brown spot |
| 14 | Leaf blast |
| 15 | Leaf scald |
| 16 | Narrow brown |
| 17 | healthy |
