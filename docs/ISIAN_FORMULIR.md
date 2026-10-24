# Isian formulir submit — hasil final

Angka di bawah berasal dari evaluasi bobot final yang hash-nya sudah diperiksa. Tautan masih berupa template dan baru tersedia setelah pemilik akun mengunggah paket.

| Kolom | Nilai final |
|---|---|
| Model yang disubmit | `yolo26s_24b16` |
| Link Model AI (GitHub Release) | `https://github.com/USERNAME/REPO/releases/tag/v1.0.0` |
| Link Jupyter Notebook (.ipynb GitHub Release) | `https://github.com/USERNAME/REPO/releases/download/v1.0.0/Smart_Agriculture_Experiments.ipynb` |
| Link Repositori GitHub | `https://github.com/USERNAME/REPO` |
| mAP@50 (%) | **`66.80`** |
| F1-Score (%) | **`67.31`** |

Semua pola URL di atas adalah **template, bukan tautan publik yang telah dibuat**. Ganti akun, repositori, dan tag sesuai halaman GitHub yang sebenarnya. Notebook dapat pula ditautkan melalui `https://github.com/USERNAME/REPO/blob/main/Smart_Agriculture_Experiments.ipynb` jika formulir menerima tautan Repository.

Catatan evaluasi yang dapat ditempel pada deskripsi:

> Evaluasi object detection pada test internal berisi 1.059 gambar dan 18 kelas. mAP@50 menggunakan IoU 0,50. F1-Score adalah macro-F1, yaitu rata-rata F1 per kelas pada IoU 0,50, dengan confidence `0.26626626626626626` yang dipilih dari validation. Checkpoint dipilih berdasarkan validation mAP@50–95. Test internal ini juga telah digunakan untuk membandingkan eksperimen sebelumnya; hasil ini belum merupakan evaluasi lapangan independen.

Bukti berkas final:

- Bobot: `package/release_assets/best.pt`
- Metrik: `package/release_assets/metrics.json`
- SHA256 bobot: `2b4a698f807590bea2a8f91bcd9d7b54b52f3b48d76640a421fb9fb9398d959b`
- Notebook: `package/release_assets/Smart_Agriculture_Experiments.ipynb`

Angka di atas harus berasal dari `test.mAP50 * 100` dan `test.macro_f1_iou50 * 100` pada JSON yang sama. Precision, recall, mAP@50–95, dan classification accuracy bukan pengganti dua kolom yang diminta.
