# Smart Agriculture — yolo26s_24b16

Bobot object detection untuk 18 label tanaman/kondisi daun dari dataset Telepati. Rilis ini memuat kode eksperimen, notebook, bobot terpilih, dan bukti evaluasi yang cocok dengan hash model.

| Evaluasi test internal | Persentase |
|---|---:|
| mAP@50 | 66.80% |
| mAP@50–95 | 47.25% |
| F1-Score (macro, IoU 0,50) | 67.31% |

Checkpoint dipilih dari validation mAP@50–95. Confidence untuk F1 dipilih dari validation: `0.26626626626626626`. F1 adalah rata-rata F1 per kelas. Test berisi 1.059 gambar; benchmark internal ini juga digunakan dalam eksperimen sebelumnya. Hasil tidak menetapkan akurasi diagnosis lapangan atau kinerja drone.

SHA256 `best.pt`: `2b4a698f807590bea2a8f91bcd9d7b54b52f3b48d76640a421fb9fb9398d959b`.

Aset:

- `best.pt`: bobot model final.
- `Smart_Agriculture_Experiments.ipynb`: notebook eksperimen dan petunjuk reproduksi.
- `metrics.json`: hasil evaluasi model final.
- `MODEL_CARD.md`: penggunaan, kelas, protokol, dan keterbatasan.
- `SHA256SUMS.txt`: integritas berkas.
- `smart-agriculture-release.zip`: arsip tambahan aset rilis, jika disertakan.

Sumber eksperimen dan cara menyiapkan dataset ada pada README repositori. Dataset asli tidak dilampirkan dalam rilis ini. Perhatikan lisensi dependensi, bobot dasar, dan atribusi dataset yang tercantum dalam paket.
