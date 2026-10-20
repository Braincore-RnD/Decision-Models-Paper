# Dekontaminasi redo — issue #10 (CPU-only, 2026-10-20)

Metode: replikasi setia `Decontaminator` Unsloth (`unsloth/models/decision_datasets.py`,
`_words` = regex `\w+` lowercase, `n=13`, fallback whole-string bila `<13` kata).
Sumber = split kanonis HF yang dipakai mixture (lihat `SOURCES` di lib):
`legacy-datasets/banking77` (train 10.003 / test 3.080),
`clinc/clinc_oos:plus` (train 15.250 / test 5.500),
`LocalLLaMA/typed-decisions:all` (train 1.200 / test 400).
Near-dup = TF-IDF word-level + cosine (sklearn, CPU). Skrip: `dekon_redo.py`,
hasil mentah: `dekon_redo.json`.

## (a) Audit jujur 13-gram — klaim issue separuh benar, separuh salah

Distribusi panjang utterance (kata, tokenizer regex lib):

| Split | n | mean | median | max | < 13 kata | % |
|---|---|---|---|---|---|---|
| BANK test | 3.080 | 11,22 | 9 | 69 | 2.319 | **75,3%** |
| BANK train | 10.003 | 12,25 | 10 | 84 | 7.095 | 70,9% |
| CLINC test | 5.500 | 8,44 | 8 | 26 | 4.912 | **89,3%** |
| CLINC train | 15.250 | 8,47 | 8 | 28 | 13.583 | 89,1% |

Angka `% < 13 gram` yang diminta: **BANK test 75,3% — CLINC test 89,3%**.

Koreksi jujur terhadap issue: klaim "utterance < 13 token → tidak menghasilkan
shingle → filter tidak pernah menembak" **salah secara teknis**. Kode lib:

```python
def _shingles(self, text):
    words = _words(text)
    if len(words) < self.n:
        return {" ".join(words)} if words else set()   # fallback: whole-string exact match
```

Teks pendek tetap menghasilkan SATU shingle = seluruh utterance ternormalisasi,
sehingga **duplikat eksak tetap tertangkap**. Yang benar-benar hilang pada teks
pendek adalah *partial-match* (satu kata beda → shingle beda total → lolos).
Jadi 13-gram = **exact-match detector untuk 75–89% utterance**, bukan "tidak ada
dekontaminasi". Kelemahan sebenarnya:

1. **Parafrase/near-dup lolos total** (satu kata diganti → shingle beda).
2. **Lubang cakupan**: `eval_texts` train script hanya berisi typed full +
   banking77/clinc150 **masing-masing limit 500 (seed 3407)** — dari test
   3.080/5.500. Proxy-pengukuran: **83,7% BANK test dan 90,9% CLINC test tidak
   tercakup shingle subset-500** (aproksimasi `random.Random(3407)`; subset
   eksak memakai `datasets.shuffle(seed=3407)` sehingga angka ini ±, orde
   besarnya kokoh: 5/6 test tak pernah dilihat filter).

## (b) Redo: exact + 8-gram + TF-IDF cosine > 0,9 (train-pool vs test)

| Cek | BANK (test 3.080) | CLINC (test 5.500) |
|---|---|---|
| Exact ternormalisasi train→test | 25 (0,81%) | 19 (0,35%) |
| Berbagi ≥1 8-gram dgn train | 326 (10,58%) | 165 (3,0%) |
| TF-IDF cosine ≥ 0,90 | **502 (16,3%)** | **440 (8,0%)** |
| TF-IDF cosine ≥ 0,95 | 214 (6,95%) | 178 (3,24%) |
| TF-IDF cosine = 1,0 | 56 (1,82%) | 62 (1,13%) |
| Rata-rata max-sim per test item | 0,727 | 0,647 |

Cross-dataset (sinyal mirror antar-taksonomi): exact bank-train↔clinc-test = 2,
clinc-train↔bank-test = 5; TF-IDF ≥0,9 = 18 dan 17. **Tidak ada mirror
sistematis** — taksonomi BANK vs CLINC memang disjoint secara leksikal.
Typed train↔test: 0 duplikat state.

Estimasi baris train yang akan dibuang redo (vs subset-500 dekon aktual):
BANK-train 10.003 → exact 7 (0,07%), 8-gram 96 (0,96%), TF-IDF≥0,9 **139
(1,39%)**; CLINC-train 15.250 → exact 0, 8-gram 28 (0,18%), TF-IDF≥0,9 **56
(0,37%)**. Total ~195 baris intent-train (~0,8% pool intent) + proporsional
kecil dari 10 sumber non-intent lain (tak diukur; utterance panjang → 13-gram
sudah memadai di sana).

## (c) Audit LOTO eksplisit

LOTO di `unsloth_kaggle_dec.py:79-80` mengeluarkan **seluruh sumber taksonomi
target dari `MIX_SOURCES`** (bukan hanya test split) — train LOTO-banking tidak
pernah melihat `banking77` train MAUPUN test. Sisa risiko LOTO = kebocoran via
sumber lain: cross-dataset exact 2–5 item + TF-IDF≥0,9 17–18 item per arah
(≤0,6% test). Kecil, bukan penjelas transfer 43,4/55,6.

## Keputusan: TIDAK retrain

Alasan: (1) duplikat eksak sudah tertangkap fallback whole-string (25+19 item
itu *diketahui tertangkap* filter lama, bukan temuan baru); (2) tambahan
temuan redo hanya ~195 baris train (~0,8% pool intent, ~0,65% mixture 30k);
(3) LOTO memutus sumber target total sehingga temuan ini tak menyentuh angka
transfer headline; (4) biaya retrain (kuota GPU ~habis, 27,29/30h) tak
proporsional dengan efek yang diharapkan (<1% data). Tindak lanjut murah:
tulis limitasi ini di naskah + rencanakan filter 8-gram + TF-IDF≥0,9 di
protokol berikutnya (1 baris config bila lib mendukung `n` custom — lib sudah
mendukung `Decontaminator(eval_texts, n=...)`, tinggal pass `n=8` + pra-filter
TF-IDF offline).

## Paragraf siap-tempel buat naskah (JANGAN edit paper/*.tex langsung)

> Decontamination uses Unsloth's 13-gram shingle filter against the canonical
> test splits (typed-decisions full test plus 500-item subsamples, seed 3407,
> of BANKING77 and CLINC150). We disclose two limitations. First, 75.3% of
> BANKING77 and 89.3% of CLINC150 test utterances are shorter than 13 words,
> where the filter degrades to whole-string exact matching (verified against
> the library source): paraphrases and near-duplicates pass through. A
> post-hoc TF-IDF audit (cosine ≥ 0.9, CPU) finds 502/3,080 (16.3%) BANKING77
> and 440/5,500 (8.0%) CLINC150 test utterances with a near-duplicate in the
> corresponding train pool, and cross-taxonomy leakage is negligible (≤18
> items per direction). Second, only 500 of 3,080/5,500 test items enter the
> filter, leaving ~84–91% of test shingles uncovered. LOTO runs remove the
> held-out taxonomy's source entirely (train and test), so the residual
> leakage (~0.8% of intent-train rows) cannot explain the transfer numbers;
> we did not retrain and instead report this audit as a bound.
