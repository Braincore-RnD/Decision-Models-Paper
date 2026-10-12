Menurut saya, ide penelitian kamu applicable, dan sudah punya fondasi yang cukup konkret untuk dikembangkan menjadi paper ilmiah. Bahkan, berdasarkan outline yang kamu lampirkan, arah risetnya sudah lebih kuat daripada sekadar “melakukan SFT pada LLM kecil”.

Saya melihat kontribusi utamanya bukan menciptakan arsitektur model baru, melainkan menyelidiki bagaimana strategi pelatihan dan komposisi data memengaruhi kemampuan transfer, kualitas keputusan, serta kalibrasi probabilitas pada model keputusan berukuran kecil.

Baseline kamu, [itsdevice/lfm-2.5-350m-decs1 di Hugging Face](https://huggingface.co/itsdevice/lfm-2.5-350m-decs1), juga memberi titik awal yang jelas untuk penelitian empiris. Kita bisa memperlakukan model tersebut sebagai artefak awal, lalu membangun eksperimen terkontrol di atasnya.

Ada satu hal yang menurut saya paling menarik: outline kamu sudah memuat klaim bahwa schema augmentation meningkatkan hasil BANKING77 dari 36% menjadi 83%, dan CLINC150 dari 33% menjadi 73%. Kalau hasil ini bertahan pada pengujian yang lebih ketat, kontribusi penelitianmu bukan sekadar “model kecil bisa di-fine-tune”, melainkan bahwa cara kita menyusun data pelatihan mungkin sama pentingnya dengan pilihan model dan ukuran parameter.

Namun, angka tersebut saat ini harus diperlakukan sebagai hasil awal dari eksperimen kamu, bukan hasil yang sudah terverifikasi secara independen. Saya akan membedakan temuan yang sudah kamu laporkan, hal yang masih perlu diuji, dan klaim yang dapat dipertahankan dalam paper.

## 1. Seberapa kuat potensi penelitian ini?

Saya menilainya dari empat aspek berikut.

Relevansi topik: kuat

Model keputusan non-generatif adalah area yang sedang berkembang. Pada 8 Oktober 2026, preprint [TypedBench](https://arxiv.org/abs/2610.11392) mengusulkan evaluasi yang mencakup akurasi, sensitivitas terhadap perubahan wording, kalibrasi, dan biaya keputusan. Ini sangat dekat dengan arah evaluasi yang kamu rencanakan.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

Kelayakan teknis: kuat

Menurut saya, ide penelitian kamu applicable, dan sudah punya fondasi yang cukup konkret untuk dikembangkan menjadi paper ilmiah. Bahkan, berdasarkan outline yang kamu lampirkan, arah risetnya sudah lebih kuat daripada sekadar “melakukan SFT pada LLM kecil”.

Saya melihat kontribusi utamanya bukan menciptakan arsitektur model baru, melainkan menyelidiki bagaimana strategi pelatihan dan komposisi data memengaruhi kemampuan transfer, kualitas keputusan, serta kalibrasi probabilitas pada model keputusan berukuran kecil.

Baseline kamu, [itsdevice/lfm-2.5-350m-decs1 di Hugging Face](https://huggingface.co/itsdevice/lfm-2.5-350m-decs1), juga memberi titik awal yang jelas untuk penelitian empiris. Kita bisa memperlakukan model tersebut sebagai artefak awal, lalu membangun eksperimen terkontrol di atasnya.

Ada satu hal yang menurut saya paling menarik: outline kamu sudah memuat klaim bahwa schema augmentation meningkatkan hasil BANKING77 dari 36% menjadi 83%, dan CLINC150 dari 33% menjadi 73%. Kalau hasil ini bertahan pada pengujian yang lebih ketat, kontribusi penelitianmu bukan sekadar “model kecil bisa di-fine-tune”, melainkan bahwa cara kita menyusun data pelatihan mungkin sama pentingnya dengan pilihan model dan ukuran parameter.

Namun, angka tersebut saat ini harus diperlakukan sebagai hasil awal dari eksperimen kamu, bukan hasil yang sudah terverifikasi secara independen. Saya akan membedakan temuan yang sudah kamu laporkan, hal yang masih perlu diuji, dan klaim yang dapat dipertahankan dalam paper.

## 1. Seberapa kuat potensi penelitian ini?

Saya menilainya dari empat aspek berikut.

Relevansi topik: kuat

Model keputusan non-generatif adalah area yang sedang berkembang. Pada 8 Oktober 2026, preprint [TypedBench](https://arxiv.org/abs/2610.11392) mengusulkan evaluasi yang mencakup akurasi, sensitivitas terhadap perubahan wording, kalibrasi, dan biaya keputusan. Ini sangat dekat dengan arah evaluasi yang kamu rencanakan.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

Kelayakan teknis: kuat

Baseline kamu sudah menggunakan backbone 350M, QLoRA, dan decision head khusus. Model card mendokumentasikan proses training dan evaluasi awal, sehingga kamu tidak harus memulai eksperimen dari nol.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

Kekuatan novelty: menjanjikan, belum final

Schema augmentation dan komposisi data dapat menjadi kontribusi ilmiah. Akan tetapi, kamu harus membuktikan bahwa keduanya menghasilkan efek yang konsisten, bukan sekadar kebetulan dari satu eksperimen atau perubahan jumlah token pelatihan.

Kesiapan publikasi: perlu eksperimen tambahan

Outline kamu sendiri sudah mengidentifikasi kebutuhan multi-seed, ablation, dan evaluasi yang lebih luas. Menurut saya, ini memang komponen yang akan menentukan apakah naskahnya terbaca sebagai penelitian yang kuat atau hanya laporan hasil fine-tuning.

## 2. Tiga klaim utama yang layak dijadikan kontribusi

Dari outline What Makes Small Decision Models Transfer?, saya akan mempertahankan dua pertanyaan utama dan menjadikan kalibrasi sebagai dimensi evaluasi lintas eksperimen.

### RQ1. Apakah schema augmentation meningkatkan transfer ke skema keputusan yang belum pernah dilihat?

Ini menurut saya kontribusi yang paling menarik.

Misalnya, model dilatih menggunakan skema berikut:

```
{
  "state": "The customer requests a refund for a duplicate payment.",
  "question": "Which team should handle this?",
  "options": [
    "billing",
    "technical",
    "sales"
  ]
}
```

Kemudian kamu melakukan beberapa transformasi pada skemanya: mengganti nama pilihan, memparafrasekan deskripsi pilihan, mengubah redaksi pertanyaan, atau menghasilkan pertanyaan turunan.

Hipotesisnya, model tidak hanya menghafal hubungan antara kalimat dan label, tetapi belajar menggunakan konteks serta deskripsi pilihan ketika membuat keputusan.

Untuk membuktikannya, saya menyarankan ablation berikut.

| Eksperimen | Schema augmentation                     |
| ---------- | --------------------------------------- |
| A          | Tidak ada, hanya skema kanonis          |
| B          | Hanya mengganti nama pilihan            |
| C          | Hanya memparafrasekan deskripsi pilihan |
| D          | Hanya menghasilkan variasi pertanyaan   |
| E          | Menggabungkan seluruh augmentasi        |

Semua eksperimen harus menggunakan pembagian data, checkpoint awal, dan anggaran pelatihan yang sebanding.

Koreksi metodologis paling penting: jika BANKING77 dan CLINC150 tetap menjadi sumber data training, performa pada test split keduanya belum menunjukkan zero-shot taxonomy transfer. Model sudah memiliki kesempatan mempelajari kedua task tersebut. Model card kamu memang mencantumkan kedua dataset di dalam campuran training, sekalipun test set-nya dideskripsikan sebagai data yang telah didekontaminasi.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

Untuk menguji transfer ke taksonomi yang belum dikenal, kamu bisa membuat eksperimen leave-one-taxonomy-out. Misalnya, seluruh contoh BANKING77 dikeluarkan dari training untuk eksperimen tersebut, lalu model diuji pada BANKING77. Ulangi dengan CLINC150. Pastikan label, definisi pilihan, serta variasi augmentasi dari task yang ditahan juga tidak bocor ke training.

Jika akurasi meningkat pada evaluasi ini, klaim transfer akan jauh lebih kuat.

### RQ2. Bagaimana komposisi dataset memengaruhi kemampuan keputusan pada kapasitas model yang tetap?

Klaim kedua dalam outline kamu menyatakan bahwa komposisi campuran data menciptakan trade-off: memperkuat satu kelompok task dapat menurunkan performa pada kelompok lain.

Ini masuk akal sebagai hipotesis penelitian. Model 350M memang memiliki kapasitas yang jauh lebih terbatas dibandingkan model besar, sehingga distribusi contoh dan tujuan pelatihan dapat memengaruhi kemampuan yang dipelajari.

Namun, saya akan mengganti klaim “zero-sum” menjadi “performance trade-offs under a fixed training budget” sampai ada bukti yang cukup untuk menyatakan sifat zero-sum secara formal.

Rancangan eksperimen yang saya sarankan:

| Eksperimen | Proporsi data keputusan terstruktur | Data task lain |
| ---------- | ----------------------------------- | -------------- |
| M1         | Rendah                              | Tinggi         |
| M2         | Sedang                              | Sedang         |
| M3         | Tinggi                              | Rendah         |
| M4         | Sangat tinggi                       | Sangat rendah  |

Persentase pastinya ditetapkan setelah mengetahui distribusi dataset dan anggaran pelatihan yang tersedia.

Jaga jumlah token pelatihan atau langkah optimizer tetap sebanding. Ini penting karena augmentasi dapat memperbesar jumlah contoh maupun panjang urutan. Jika M3 lebih unggul, kita harus tahu apakah penyebabnya komposisi data atau sekadar tambahan token pelatihan.

Ukur hasil pada setiap kelompok task secara terpisah. Kemudian gambarkan Pareto frontier antara performa in-domain, performa OOD, dan kalibrasi. Dengan cara ini, hasil penelitianmu bukan sekadar tabel akurasi, tetapi menjelaskan trade-off yang dapat dipakai peneliti lain ketika memilih resep training.

### RQ3. Apakah model keputusan kecil tetap menghasilkan probabilitas yang berguna ketika distribusinya berubah?

Bagian kalibrasi perlu dipertahankan, tetapi klaim bahwa soft atau consensus labels adalah missing ingredient masih perlu dibuktikan melalui ablation.

Jika soft labels hanya tersedia pada satu sumber data tertentu, efeknya akan tercampur dengan karakteristik sumber tersebut. Bandingkan hard labels dan soft labels pada contoh dan skema yang sama agar pengaruhnya dapat dipisahkan.

Saya menyarankan setidaknya empat metrik probabilitas:

| Metrik                           | Yang diukur                                                                             |
| -------------------------------- | --------------------------------------------------------------------------------------- |
| Brier score                      | Kualitas probabilitas terhadap hasil aktual                                             |
| Negative log-likelihood          | Seberapa baik distribusi probabilitas memberikan skor terhadap label benar              |
| Expected Calibration Error (ECE) | Selisih rata-rata confidence dan akurasi pada kelompok prediksi                         |
| Selective risk / coverage        | Risiko kesalahan ketika model hanya mengambil keputusan pada tingkat keyakinan tertentu |

ECE sebesar 0,083 pada model card adalah temuan awal yang berguna, tetapi tidak dengan sendirinya membuktikan model telah terkalibrasi dengan baik. Evaluasi harus menggunakan calibration set yang terpisah dari test set, disertai pemeriksaan pada distribusi yang bergeser.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

+1

Paper [System Switch](https://arxiv.org/abs/2610.09683), yang diajukan pada 7 Oktober 2026, juga menekankan bahwa akurasi, kalibrasi, dan kemampuan confidence dalam membedakan prediksi benar dari salah merupakan hal yang berbeda. Karena itu, evaluasi sensitivitas confidence atau AUROC for error detection akan berguna jika kamu ingin mengklaim model ini dapat dipakai untuk keputusan otomatis dengan mekanisme eskalasi.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

## 3. Baseline apa yang harus dibandingkan?

Saya tidak menyarankan kamu langsung menambahkan banyak arsitektur. Lebih baik menggunakan beberapa baseline dengan fungsi yang jelas.

Baseline A: LFM2.5-350M sebelum dan sesudah pelatihan keputusan

Ini adalah pembanding utama. Terapkan evaluasi yang identik sebelum dan sesudah adaptasi untuk mengukur perubahan yang benar-benar terjadi.

Baseline B: SFT tanpa schema augmentation

Ini eksperimen kontrol paling penting. Tanpanya, peningkatan performa tidak dapat diatribusikan dengan jelas kepada augmentasi.

Baseline C: model kecil lain dan model generatif

Gunakan satu model sekelas sekitar 350M sampai 400M yang dilatih dengan resep sebanding, serta satu baseline generatif yang menerima pertanyaan dan pilihan yang sama. Qwen3.5-0.8B dapat menjadi pembanding tambahan, tetapi hasilnya tidak boleh dianggap sebagai kontrol yang cocok dari sisi ukuran parameter.

Baseline D: biaya inferensi

Ukur latensi, throughput, penggunaan memori, dan perubahan latensi saat jumlah pertanyaan atau pilihan bertambah. Satu forward pass dan tidak adanya token generatif merupakan karakteristik sistem, tetapi keunggulan efisiensinya tetap perlu diukur secara empiris.

Sebagai referensi eksternal, panduan Unsloth melaporkan Qwen3.5-0.8B dengan skor 73% pada typed-decisions, 74% pada BANKING77, dan 76% pada CLINC150 menggunakan protokol mereka sendiri. Angka ini bisa menjadi referensi konteks, tetapi bukan pengganti perbandingan langsung dengan checkpoint, data, dan prosedur evaluasi yang sama.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://unsloth.ai\&sz=32)

Unsloth - Train and Run Models Locally

## 4. Hal yang harus dibereskan sebelum submit

Ada tiga pekerjaan prioritas.

Pertama, rekonsiliasi seluruh versi eksperimen. Model card publik menyebut campuran 12.000 baris dari 13 sumber dengan hasil 68% pada BANKING77 dan 68% pada CLINC150. Outline menyebut perubahan v5 ke v6 yang mencapai 83% dan 73%, serta eksperimen dengan campuran 30.000 baris. Keduanya bisa saja merupakan eksperimen berbeda, tetapi naskah harus menjelaskan checkpoint, dataset, pembagian data, konfigurasi training, dan evaluasi yang menghasilkan setiap angka. Jangan menggabungkan angka lintas versi menjadi satu klaim seolah-olah semuanya berasal dari protokol yang sama.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

Kedua, jalankan minimal tiga seed training dan laporkan interval kepercayaan. Untuk augmentasi dan komposisi data, eksperimen harus memiliki kontrol yang sebanding. Sebaiknya gunakan bootstrap confidence intervals atas contoh uji, bukan hanya membandingkan angka akurasi tunggal.

Ketiga, pastikan setiap transformasi data dilakukan setelah pembagian train, validation, calibration, dan test. Decontamination terhadap contoh identik saja tidak cukup jika paraphrase atau pertanyaan turunan dari contoh test masuk ke training.

Saya juga akan menggunakan istilah typed decision model atau System-1-style decision model dengan definisi operasional yang jelas. Jangan mengklaim model tersebut benar-benar mereplikasi proses kognitif System 1 pada manusia. Yang diuji di sini adalah inferensi keputusan terstruktur, bukan mekanisme psikologis manusia.

## 5. Bagaimana bentuk paper final yang saya sarankan?

Jika eksperimen transfer taksonomi berhasil, judul saat ini sudah cukup kuat. Saya hanya akan membuatnya lebih presisi:

Judul utama yang direkomendasikan

## What Makes Small Decision Models Transfer? Schema Augmentation and Data-Mixture Trade-offs in a 350M Decision Model

Fokus: efek schema augmentation, komposisi data, dan generalisasi OOD pada backbone 350M dengan decision head.

Struktur paper-nya dapat dibuat sebagai berikut:

1. Introduction. Masalah adaptasi SLM menjadi model keputusan terstruktur, keterbatasan evaluasi yang hanya berfokus pada akurasi, serta dua hipotesis utama.
2. Related Work. Decision models, supervised adaptation, schema and prompt robustness, calibration, dan efisiensi model kecil.
3. Methodology. Backbone dan decision head, format data, augmentasi, komposisi campuran data, prosedur split, serta protokol eksperimen.
4. Experimental Results. Ablation schema augmentation, eksperimen komposisi data, evaluasi taxonomy-held-out, kalibrasi, robustness, dan biaya inferensi.
5. Discussion. Kapan augmentasi berhasil, kapan terjadi trade-off, serta keterbatasan generalisasi dan kapasitas model.
6. Conclusion. Temuan yang dapat direplikasi, artefak, dataset atau skrip evaluasi, dan implikasi praktis.

Untuk posisi literatur terbaru, TypedBench dapat kamu gunakan sebagai acuan desain evaluasi, sedangkan System Switch membantu memperjelas perbedaan akurasi dan kualitas confidence. Keduanya masih berupa preprint yang baru diajukan pada Oktober 2026, sehingga status publikasi dan versinya perlu diperbarui ketika naskahmu disiapkan.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1

Kesimpulan saya: lanjutkan ide ini. Fondasi model, hipotesis tentang augmentasi, dan persoalan trade-off komposisi data sudah layak dikembangkan menjadi penelitian empiris. Hal yang paling menentukan sekarang adalah membuat klaim transfer benar-benar terukur, memisahkan pengaruh augmentasi dari pengaruh jumlah data, dan mereproduksi hasil melalui beberapa seed.

Untuk target Scopus Q3/Q4, saya akan memposisikan artikel ini sebagai empirical ML / systems research, bukan paper arsitektur baru. Dengan eksperimen yang kuat dan artefak yang dapat direproduksi, posisi ilmiahnya akan jauh lebih meyakinkan daripada sekadar melaporkan bahwa model LFM2.5-350M berhasil di-SFT menjadi decision model.

## 6. Temuan audit kelayakan Scopus Q3 (2026-10-10, berbasis RUN_LOG v1–v6 + PAPER.md)

> Status: **belum submit-ready**. 70% jalan, 5 verifikasi minimal sebelum submit.

### 6.1 Inkonsistensi yang harus direkonsiliasi dulu

1. **Versi rilis ganda**: `RUN_LOG` §1 baris 21 rilis = v2 (isi HF), verdict §1 baris 45 rilis = v3 (ECE 0.052). Paper klaim v6 (83%/73%). Satu naskah tidak boleh mencampur tiga checkpoint seolah satu protokol.
2. **Holdout tidak comparable**: v2 = 72% (mixture holdout) vs v3–v6 = 55–63%* (rekonstruksi typed-only). Footnote sudah jujur, tapi tabel utama tetap disandingkan — reviewer sebut misleading.
3. **v4–v6 tanpa kronologi kernel**: tabel hasil ada, kronologi §2 berhenti di v3 + push. Tidak reproducible.
4. **ECE ganda**: 0.052 (PAPER/RUN_LOG v3) vs 0.083 (model card HF). Tentukan mana yang dari cal-set terpisah.
5. **Latency & coverage tanpa sumber**: `sub-45ms`, `Coverage@Risk 64.2%` ada di PAPER §6, tidak ada di RUN_LOG. Ukur di T4 atau turunkan klaim.

### 6.2 Blocker metodologis (alasan desk-reject Q3)

1. **Bukan zero-shot**: BANKING77/CLINC150 ikut di train mixture (13 sumber), eval di test split yang sama = in-distribution + dekon 13-gram. Klaim C1 harus turun ke "cross-split transfer" sampai ada run leave-one-taxonomy-out.
2. **Single seed + baseline vendor**: semua run seed-3407; Qwen 73/74/76% = angka Unsloth docs, bukan rerun protokol identik. Butuh ≥3 seeds + bootstrap CI + rerun Qwen sendiri.
3. **Augmentasi tercampur token budget**: v5 34.8k → v6 43.1k decisions. Kenaikan 36%→83% bisa dari tambah token, bukan augmentasi. Butuh kontrol token-matched + ablasi rename vs paraphrase vs derived terpisah.

### 6.3 Checklist minimal sebelum submit (lazy path — v7 script-ready, run GPU pending)

- [ ] LOTO BANKING77 / CLINC150 (2 train): `HOLDOUT_TAXONOMY=banking77|clinc150`, transfer beneran?
- [ ] Rerun Qwen3.5-0.8B protokol identik (eval v7: `MODEL_ID`+`FINAL` diganti) + 3 seeds resep v6 (`SEED=3407/41/42`) + bootstrap CI (`N_BOOT`)
- [ ] Ablasi augmentasi token-matched: `AUG_MODE=none|full` + `MAX_DECISIONS` sama (A vs E dulu; B/C/D dicatat sbg full — batas lib, lihat header train v7)
- [ ] Kalibrasi: cal-set terpisah dari test (split 3-way v7, `calibrate` full dict); Brier/NLL/AUROC + uji distribusi geser = tahap dua → berjalan #16-17 (`predict` API ditemukan via DUMP_API, dump per-item + hitung lokal)
- [ ] Split train/cal/test dikunci sebelum build (`SEED`, `SEED+5`); C5 (soft labels) turun ke future work

### 6.4 Run plan v7 — tracker status (update tiap kernel jalan)

Aturan per kernel: `FORCE_NEW=1`, output/HF terpisah, smoke `MAX_STEPS=60` dulu.
Eval per kernel: eval v7 dengan `TRAIN_OUT` = out-u train; tempel `run_meta.json` +
`eval_results.json` ke `RUN_LOG.md`. Status: `belum` → `ongoing` → `selesai`.

| # | Kernel | Env kunci | Status | Kernel link | Hasil |
|---|---|---|---|---|---|
| 1 | v6-resep seed 1 | `SEED=3407` | selesai (3ep full 2.25 jam, train_loss 0.505, eval_loss 0.842; cal 74.5%/ECE 0.080 n=140; internal 70.7%/0.082 n=141; typed-ext 55.9%/0.060 n=400; 43171 decisions, skip 809) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-seed3407-full | ckpt+final+merged+meta tersimpan di output kernel |
| 2 | v6-resep seed 2 | `SEED=41` | selesai (3ep, overfit ep3 0.72→0.90 best=ep2; cal 74.4%/ECE 0.094 n=132; internal 79.9%/0.091 n=132; typed-ext 53.3%/0.079 n=400; 43172 decisions, skip 810) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-seed41-full | ckpt+final+merged+meta di output kernel |
| 3 | v6-resep seed 3 | `SEED=42` | selesai (3ep, overfit ep3 0.62→1.01 best=ep2; cal 80.0%/ECE 0.074; internal 79.0%/0.048 n=136; typed-ext 55.5%/0.049 n=400) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-seed42-full | ckpt+final+merged+meta di output kernel. 3-seed typed-ext: 55.9/53.3/55.5 (mean 54.9) |
| 4 | LOTO banking77 | `HOLDOUT_TAXONOMY=banking77` | selesai train + eval (BANKING77 transfer **44% [38–47%]** vs chance 2%, para-median 42.8% range [42.2–43.6]; CLINC in-train 65% [58–68]; typed 59% [56–61]; latensi terukur 64–249ms/item → klaim sub-45ms direvisi di tahap paper) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-loto-banking-full | C1 = transfer substansial, bukan 83% in-distribution |
| 5 | LOTO clinc150 | `HOLDOUT_TAXONOMY=clinc150` | selesai (3ep, best ep2 0.64; cal 78.0%/ECE 0.049; internal 79.0%/0.070 n=139; typed-ext 57.6%/0.082 n=400) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-loto-clinc-full | ckpt+final+meta di output kernel |
| 9 | Eval LOTO banking (transfer beneran?) | eval v7 `TRAIN_OUT`=run #4, `N_BOOT=30` | selesai (tabel: typed 31→59% [56–61], BANKING77 2→44% [38–47], CLINC 17→65% [58–68], holdout 80% train-split) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-eval-loto-banking | `eval_results.json` di output kernel |
| 12 | Eval LOTO clinc (pasangan C1) | Colab T4 `jev-eval2` (CPU gagal: unsloth GPU-only; kernel GPU dihapus pre-run) `N_BOOT=0 PARA=1 DUMP_API=1` → selesai (eval: typed 57.6%, BANKING77 in-train 82.2% para [81.8–82.6], CLINC150 unseen **55.6%** para [55.6–57.0]; holdout 79% loto=clinc150; cal ECE 0.049; latensi 56/124/220ms) | selesai | arsip docs/eval/colab/loto-clinc/ (eval_results.json + eval.log, token-free) |
| 13–15 | Eval 3 seed (variansi BANKING77/CLINC150 tabel utama, test identik) | selesai semua. In-dist 3-seed: typed 55.9/53.3/55.6 (mean 54.9); BANK 76.8/83.4/83.8 (mean 81.3); CLINC 68.2/76.6/73.8 (mean 72.9). #13 seed3407 Kaggle-lean, #14 seed41 Kaggle-lean, #15 seed42 Colab | selesai | arsip docs/eval/kaggle/seed3407/ + seed41/ + docs/eval/colab/seed42/ |
| 6 | Ablasi A (kanonis) | `AUG_MODE=none` + `MAX_DECISIONS=28000` | selesai train + eval (eval: typed 54% [51–56], BANKING77 **86% [81–88]**, CLINC150 **84% [81–88]**; 28000 rows→31754 decisions) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-ablasi-none-full | ckpt+final+meta di output kernel |
| 7 | Ablasi E (full) | `AUG_MODE=full` + `MAX_DECISIONS=28000` (rows sama dgn #6; decisions E 39560 vs A 31754 (+25% E) — bias menguntungkan E, hasil memenangkan A = konservatif; confound kedua: batch 4 vs 8, efektif sama 32) | selesai train + eval (eval: typed 59% [56–61], BANKING77 **79% [75–82]**, CLINC150 **70% [65–73]**) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-ablasi-full-full | ckpt+final+meta di output kernel |
| 10 | Eval ablasi A (kanonis) | eval v7 `TRAIN_OUT`=run #6, `N_BOOT=30` | selesai (tabel di atas; holdout 80% train-split aug=none) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-eval-ablasi-none | `eval_results.json` di output kernel |
| 11 | Eval ablasi E (full) | eval v7 `TRAIN_OUT`=run #7, `N_BOOT=30` | selesai (tabel di atas; holdout 82% train-split aug=full) | https://www.kaggle.com/code/amalanfadilgaib/lfm-decs1-v7-eval-ablasi-full | `eval_results.json` di output kernel |
| 8 | Qwen baseline (terpisah, bukan vN) | DROPPED — pakai angka vendor Unsloth (73/74/76%) sbg konteks, label wajib "vendor-reported, beda protokol & ukuran (0.8B vs 350M)". Konsekuensi: tidak boleh ada klaim "mengalahkan/sebanding Qwen"; klaim utama tetap dari eksperimen terkontrol #1–7 + konteks d1-omni-600M (587M, satu lineage) | – | – |

Tahap dua (setelah #1–8 hijau): B/C/D terpisah, Brier/NLL/AUROC error-detection,
Decision Index slice BFCL/ToolRet, uji distribusi geser.

| # | Kernel | Env kunci | Status | Kernel link | Hasil |
|---|---|---|---|---|---|
| 16 | Recon `predict` API (Colab T4, tanpa train) | selesai: `predict(model, tok, state, questions)->{answer+probabilities terkalibrasi}`. Infra: `predict_dump.py` + `score_preds.py` (terverifikasi sintetis) | selesai | sesi ephemeral, arsip kode di repo |
| 17 | Dump + metrik RQ3 (Colab T4) | selesai dua model. LOTO-banking (shift): BANK unseen ECE **0.366**/Brier 0.876/cov 0.6% vs CLINC in-train ECE 0.176/cov 29% — kalibrasi tak transfer, tapi AUROC 0.73 (daya bedakan bertahan). E (in-dist): BANK ECE **0.026**/cov 63%, CLINC ECE 0.09/cov 40%. Klaim coverage 64.2% PAPER mati (max 63% in-dist, ~0% di shift). | selesai | arsip docs/eval/colab/loto-banking/ + ablasi-full/ (items_pred.json + metrics*.json, token-free) |
