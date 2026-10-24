# Diagnosis data dan model sesuai arahan dosen

Sumber metrik: `reports/evaluation/metrics.json`. Analisis perbaikan menggunakan **validation**, bukan test untuk memilih percobaan baru.

## 1. AP per kelas dan confusion matrix

| Kelas | Gambar train | Kotak efektif | AP50 val | AP50–95 val | Recall val | Kotak <32² px |
|---|---:|---:|---:|---:|---:|---:|
| Brown spot | 259 | 1622 | 31.01% | 13.48% | 28.12% | 60.60% |
| BrownSpot | 856 | 3202 | 34.46% | 14.57% | 27.96% | 42.72% |
| Bacterial leaf blight | 234 | 476 | 37.21% | 16.05% | 42.35% | 6.30% |
| Leaf Scald | 529 | 1024 | 43.84% | 23.24% | 38.99% | 16.31% |
| Blast | 827 | 2987 | 51.66% | 27.26% | 43.01% | 30.63% |
| Sheath Blight | 399 | 1114 | 52.78% | 30.57% | 57.17% | 17.41% |
| Leaf scald | 147 | 147 | 51.47% | 31.01% | 98.04% | 0.00% |
| Leaf blast | 262 | 283 | 50.47% | 37.83% | 47.32% | 3.89% |
| Rice-Tungro | 765 | 2468 | 65.25% | 40.48% | 60.22% | 13.25% |
| healthy | 271 | 272 | 48.69% | 42.11% | 86.11% | 0.37% |
| Healthy | 349 | 444 | 65.33% | 50.55% | 78.11% | 2.93% |
| Narrow brown | 222 | 222 | 99.29% | 53.74% | 98.07% | 0.00% |
| Bacterial panicle Blight | 443 | 450 | 83.32% | 58.09% | 84.44% | 1.78% |
| Infected Blast | 681 | 689 | 86.40% | 62.49% | 85.87% | 1.16% |
| False-Smut | 717 | 733 | 94.02% | 64.96% | 94.19% | 1.36% |
| Healthy Rice beads | 552 | 590 | 94.14% | 67.64% | 93.06% | 5.25% |
| Leaf-roller | 676 | 683 | 93.45% | 70.06% | 89.11% | 0.00% |
| Healthy Rice Leaf | 592 | 596 | 89.83% | 78.04% | 94.87% | 0.00% |

Matriks bawaan evaluasi AP memakai confidence0.001 dan IoU pencocokan0.45. Itu bukan matriks pada threshold operasional. Matriks operasional validation dibuat terpisah pada confidence pilihan F1 dan IoU0.50; sumbu baris=prediksi, kolom=label sebenarnya, dengan background untuk FP/FN.

## 2. Kualitas label

Ditemukan **2 kelompok gambar dengan piksel identik tetapi hash anotasi berbeda**. Salah satunya menunjukkan Leaf Scald dengan tiga kotak gejala versus Leaf scald dengan satu kotak daun. Anggota train yang bertabrakan dengan test telah dikeluarkan dari train; tidak ada kebocoran lintas split yang dipertahankan melalui kelompok itu. Perbedaan lain perlu diperiksa berdasarkan koordinat aktual, bukan dianggap otomatis salah.

Pasangan nama yang perlu kebijakan anotasi: BrownSpot/Brown spot; Healthy/healthy; Leaf Scald/Leaf scald. Nama mirip tidak cukup untuk menggabungkan kelas. Tentukan dahulu objek target (lesi/daun/bulir), definisi penyakit, dan kebijakan satu atau banyak kotak.

## 3. Keseimbangan dan penambahan data

Rasio gambar kelas terbanyak/tersedikit **5.82×**, rasio kotak **21.78×**. Gambar, jumlah sumber/sesi, dan kepadatan kotak perlu dipisahkan: banyak lesi pada sedikit foto bukan keragaman sampel yang tinggi.

`Brown spot` memiliki1.622 baris kotak train pada259 gambar; rendahnya AP bukan semata kekurangan kotak. `Leaf scald` memiliki147 gambar train, sehingga pengumpulan foto baru yang independen lebih berguna daripada sekadar menduplikasi fotonya.

Prioritaskan foto train baru dari sesi/lahan beragam untuk kelas lemah. Augmentasi dan oversampling hanya pada train, berbasis gambar/sumber, dengan batas pengulangan; pertahankan box dan warna gejala, audit hasilnya, dan jangan menambah salinan validation/test ke train. Oversampling dapat ikut menggandakan kelas lain pada foto yang sama.

## 4. Resolusi

Semua gambar tersimpan640×640. **22.59%** kotak train berluas di bawah32² piksel. Resolusi768 sudah dicoba pada fine-tune YOLO11s. Ini membesarkan input1.2× per sisi, tetapi tidak mengembalikan detail yang sudah hilang pada sumber640. Untuk gejala sangat kecil, pertimbangkan foto asli lebih tinggi atau crop/tiling beranotasi yang benar; seluruh turunan satu foto harus tetap satu split.

## 5. Threshold F1

Confidence **0.266266266** dipilih dari puncak kurva macro-F1 validation, lalu diterapkan ke test. AP dihitung sepanjang kurva confidence. F1 formulir adalah rata-rata F1 per kelas pada IoU0.50; bukan F1 yang dihitung dari rata-rata precision/recall.

## 6. Perbandingan arsitektur

YOLO26s dan YOLO11s sudah dibandingkan dengan data dan18 kelas yang sama. Tambahan epoch serta input768 membuat anggaran berbeda, sehingga gain tidak diatribusikan ke arsitektur saja. RF-DETR/D-FINE perlu uji GPU dan pengukuran validation dengan protokol yang sama sebelum dianggap lebih baik. Default resolusi RF-DETR kecil tidak boleh langsung dianggap cocok untuk lesi kecil.

Keputusan label tidak diotomatisasi dari skor test. Audit ini tidak mengubah anotasi, identitas kelas, atau data evaluasi.

## Uji resolusi terkontrol pada bobot yang sama

| Input | Val mAP50 | Val mAP50–95 | Val macro-F1 | Confidence dari val |
|---|---:|---:|---:|---:|
| 640 | 65.14% | 43.45% | 66.24% | 0.2663 |
| 768 | 63.87% | 36.66% | 64.59% | 0.2643 |
| 960 | 53.16% | 18.22% | 54.78% | 0.2833 |

Input operasional terpilih: **640** berdasarkan validation mAP50–95. Tidak ada angka test yang dipakai untuk memilih resolusi. Angka F1 pada tabel adalah validation, bukan isian test formulir.

## Kesalahan kelas pada threshold operasional

Matriks validation: confidence **0.266266**, IoU matching **0.50**, input **640**. 2381 kotak rujukan tidak terpasangkan; 1151 prediksi tidak terpasangkan. Kesalahan kelas pada kotak yang cocok dipisahkan dari kedua hitungan itu.

| Kelas sebenarnya | Diprediksi sebagai | Kotak |
|---|---|---:|
| Healthy | healthy | 45 |
| healthy | Healthy | 19 |
| Bacterial panicle Blight | Infected Blast | 14 |
| Infected Blast | Bacterial panicle Blight | 8 |
| BrownSpot | Blast | 8 |
| Rice-Tungro | Leaf Scald | 8 |
| Leaf Scald | Rice-Tungro | 8 |
| Leaf blast | Brown spot | 8 |
| Leaf-roller | Healthy Rice Leaf | 5 |
| Leaf scald | Leaf Scald | 5 |

[Confusion matrix operasional](../reports/diagnostics_operating_val/confusion_matrix_normalized.png). Pencocokan matriks dilakukan lintas kelas secara greedy berdasarkan IoU; AP/F1 resmi memakai pencocokan per kelas sepanjang kurva. Jangan mengganti F1 formulir dengan perhitungan dari matriks diagnosis ini.

Dari4.758 kotak validation, **2381 terlewat** dan **145 cocok dengan kelas keliru**. Pada konfigurasi ini, kotak terlewat adalah komponen kesalahan yang lebih besar daripada pertukaran kelas.

| Kelas | Kotak val | Terlewat | Kelas keliru |
|---|---:|---:|---:|
| BrownSpot | 1148 | 838 | 8 |
| Blast | 730 | 427 | 1 |
| Rice-Tungro | 744 | 315 | 8 |
| Brown spot | 313 | 223 | 3 |
| Sheath Blight | 425 | 190 | 0 |
| Leaf Scald | 318 | 189 | 14 |

## Kandidat dibandingkan pada protokol validation yang sama

| Model | Input | Total epoch | Val mAP50 | Val mAP50–95 | Val macro-F1 | Confidence F1 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | 640 | 12 | 61.12% | 36.19% | 60.92% | 0.2212 |
| YOLO26s | 640 | 24 | 65.14% | 43.45% | 66.24% | 0.2663 |
| YOLO11s_refine | 768 | 28 | 61.14% | 39.54% | 62.54% | 0.2322 |

Pemenang agregat validation mAP50–95: **YOLO26s**. Lihat `reports/validation_candidates/per_class_comparison.csv` untuk54 baris perbandingan per kelas. Dataset, jumlah kelas, confidence floor AP, NMS, batas deteksi, dan batch evaluasi sama; ukuran input, resep serta anggaran training berbeda dan dicatat. Ini membandingkan konfigurasi keseluruhan, bukan ablation arsitektur murni.

## RF-DETR: kelayakan telah diuji, akurasi belum dibandingkan penuh

RF-DETR Nano1.9.0 berhasil menjalankan satu epoch pilot:148 gambar train,74 validation, tanpa test. Input640, batch8, akumulasi2, gradient checkpointing. Puncak alokasi GPU3499MiB, cadangan allocator4020MiB pada RTX4050 Laptop6GB.

Estimasi12 epoch full train dari pilot sekitar5.4jam, belum termasuk validation penuh dan penulisan checkpoint; alokasi praktis sekitar6–8jam. Estimasi bukan jaminan waktu. Pengguna memilih finalisasi YOLO26s dan audit untuk tenggat submission. RF-DETR tidak diberi skor pembanding final dari pilot kecil ini.

RF-DETR/varian yang lebih besar maupun D-FINE tidak otomatis lebih baik pada lesi kecil atau label yang tidak konsisten. Untuk percobaan berikutnya, gunakan split/kelas yang sama, evaluasi per kelas, dan anggaran yang dicatat. [Sumber RF-DETR](https://github.com/roboflow/rf-detr), [sumber D-FINE](https://github.com/Peterande/D-FINE).
