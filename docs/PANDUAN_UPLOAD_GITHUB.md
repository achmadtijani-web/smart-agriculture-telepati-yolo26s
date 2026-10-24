# Berkas dan cara submit ke GitHub

Paket ini disiapkan di komputer lokal. Tautan dalam contoh berikut adalah **template**, belum menjadi tautan publik. Ganti `USERNAME` dengan nama akun GitHub dan `REPO` dengan nama repositori yang benar setelah diunggah.

## Berkas yang digunakan

| Lokasi dalam folder proyek | Penggunaan |
|---|---|
| `package/repository/` | Isi kode sumber, README, notebook, konfigurasi, dan laporan yang disiapkan untuk repositori publik. |
| `package/smart-agriculture-source.zip` | Arsip kode sumber. Ekstrak terlebih dahulu untuk unggah melalui browser. |
| `package/release_assets/best.pt` | Bobot model terpilih untuk GitHub Release. |
| `package/release_assets/Smart_Agriculture_Experiments.ipynb` | Notebook eksperimen, diunggah juga sebagai aset Release. |
| `package/release_assets/metrics.json` | Bukti metrik model yang sama dengan `best.pt`. |
| `package/release_assets/MODEL_CARD.md` | Identitas, penggunaan, batas kemampuan, dan protokol evaluasi model. |
| `package/release_assets/SHA256SUMS.txt` | Hash untuk memeriksa integritas berkas. |
| `package/smart-agriculture-release.zip` | Salinan arsip aset rilis sebagai unduhan tambahan. |
| `docs/ISIAN_FORMULIR.md` | Ringkasan angka final dan kolom tautan untuk formulir, setelah finalisasi paket. |

Gunakan nama berkas yang benar-benar terdapat di paket final. Jika hanya berkas `.template` yang tersedia, isian belum difinalisasi.

## 1. Buat repositori publik

1. Masuk ke akun GitHub, lalu buka [New repository](https://github.com/new).
2. Isi nama, misalnya `smart-agriculture-rice-detection`, dan deskripsi singkat.
3. Pilih **Public**. Buat repositori dengan **Create repository**. Jika memilih Add README, README awal tersebut dapat diganti dengan README dari paket.

Repositori publik dapat dibaca penilai tanpa izin khusus. Langkah antarmuka mengacu pada [panduan pembuatan repositori GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-a-repository-for-your-project-on-github).

## 2. Unggah kode sumber dan notebook

1. Buka folder `package/repository/`, atau ekstrak `smart-agriculture-source.zip`.
2. Pada halaman repositori pilih **Add file → Upload files**. Pada repositori kosong, gunakan tautan **uploading an existing file**.
3. Seret **isi** folder `repository`, termasuk subfoldernya. Pastikan `README.md` dan `Smart_Agriculture_Experiments.ipynb` tampil langsung pada tingkat utama repositori. Jangan hanya mengunggah ZIP atau folder pembungkus `repository`.
4. Jika jumlahnya lebih dari 100 berkas, unggah berkas tingkat utama dahulu, lalu subfolder dalam beberapa unggahan dengan struktur yang sama. Isi pesan commit, misalnya `Add reproducible agriculture detection experiment`, lalu klik **Commit changes**.
5. Klik notebook dan pastikan GitHub dapat menampilkannya. Unggah `.gitignore` dari paket juga; Windows mungkin menyembunyikan nama yang diawali titik.

Unggahan browser dibatasi 25 MiB per berkas dan 100 berkas per unggahan. Berkas lebih besar dapat ditangani lewat Git, tetapi GitHub memblokir berkas Git biasa di atas 100 MiB. Simpan bobot model sebagai aset Release. [Batas unggahan berkas GitHub](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository), [berkas besar di GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Paket sumber tidak memerlukan `.venv`, dataset asli, cache, atau seluruh folder `runs`. Dataset diperoleh dan disiapkan terpisah mengikuti README. Atribusi dataset dan lisensi dependensi tetap mengikuti catatan dalam paket.

## 3. Buat GitHub Release

1. Pada halaman repositori klik **Releases → Draft a new release**.
2. Buat tag `v1.0.0`, pilih branch sumber yang sudah diunggah, dan isi judul rilis.
3. Salin isi `release_notes.md` yang sudah berisi angka final. Berkas ini juga tersedia di dalam paket sumber. Jangan gunakan berkas template yang masih memuat penanda `{{...}}`.
4. Di bagian lampiran aset, unggah semua berkas dalam `package/release_assets/` satu per satu atau sekaligus. Pastikan **`best.pt` dan notebook `.ipynb` muncul sebagai aset tersendiri**. Tambahkan `smart-agriculture-release.zip` sebagai arsip tambahan.
5. Tunggu setiap unggahan selesai. Simpan draft jika masih memeriksa aset; setelah lengkap klik **Publish release**.

Urutan ini mengikuti [pengelolaan Release GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository?tool=webui). Setiap aset Release harus berukuran kurang dari 2 GiB. [Batas aset Release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## 4. Salin tiga tautan ke formulir

Berikut pola URL, **bukan tautan yang sudah dibuat**:

| Kolom formulir | Tautan yang disalin setelah publikasi |
|---|---|
| Link Model AI (GitHub Release) | `https://github.com/USERNAME/REPO/releases/tag/v1.0.0` |
| Link Jupyter Notebook (.ipynb GitHub Release) | `https://github.com/USERNAME/REPO/releases/download/v1.0.0/Smart_Agriculture_Experiments.ipynb` |
| Link Repositori GitHub | `https://github.com/USERNAME/REPO` |

Untuk kolom model, halaman Release menampilkan nama model, metrik, dan aset `best.pt`. Jika formulir meminta unduhan bobot langsung, salin tautan aset: `https://github.com/USERNAME/REPO/releases/download/v1.0.0/best.pt`.

Jika kolom notebook menerima Repository, alternatifnya: `https://github.com/USERNAME/REPO/blob/main/Smart_Agriculture_Experiments.ipynb`. Ganti `main` jika branch yang digunakan berbeda. Pola URL aset mengikuti contoh resmi [Release API GitHub](https://docs.github.com/en/rest/releases/releases); cara paling aman adalah menyalin tautan aset yang benar-benar tampil pada Release.

Buka ketiga tautan di jendela privat/incognito. Pastikan repo dan Release terlihat tanpa login, `best.pt` dapat diunduh, dan berkas notebook berakhiran `.ipynb`.

## 5. Isi angka evaluasi

Gunakan angka **test** pada `docs/ISIAN_FORMULIR.md`, bukan skor validation atau angka dari model lain. mAP@50 diambil dari `test.mAP50 × 100`; F1-Score dari `test.macro_f1_iou50 × 100` dalam `metrics.json` yang menyertai bobot final.

F1 yang digunakan adalah rata-rata F1 per kelas pada IoU 0,50 dan ambang confidence yang dipilih dari validation. Jangan menghitung ulang F1 dari rata-rata precision dan recall karena hasilnya tidak sama. Jika formulir meminta dua desimal, gunakan pembulatan dua desimal dan sesuaikan titik/koma dengan kolom formulir.
