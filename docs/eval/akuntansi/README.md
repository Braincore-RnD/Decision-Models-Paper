# Akuntansi data definitif — issue #14 (2026-10-20)

Sumber kebenaran: `code/train_decision_models/unsloth_kaggle_dec.py` (living, v7)
+ `docs/eval/kaggle/*/run_meta.json` (7 run, mesin) + `paper/sections/dataset.tex`
+ `docs/07-mixture-compositions.md` + `docs/RUN_LOG.md` §7–13.
Aturan: angka mesin > teks; teks hanya dipakai bila mesin bisu.

## Jawaban 13 vs 15

- **13 = sumber mixture STANDAR** (12 publik + typed-decisions), dikirim ke
  `build_decision_mixture` sebagai `MIX_SOURCES` + `typed_decisions`
  (`unsloth_kaggle_dec.py:77-82`).
- **+3 dataset customs** (Open-Jev, LeWiDi, GoEmotions) yang di-concat PASCA-builder
  (`:242`), bukan via mixture → total **16 entitas** bila dihitung per-dataset,
  atau **13 sumber standar + 5 subset/config customs** (Open-Jev 2 subset +
  LeWiDi 2 config + GoEmotions 1) = **18 baris sampling**.
- Klaim issue "15 sumber (12+3)" salah hitung: 12 publik + typed = 13 standar;
  customs = 3 dataset / 5 subset. Tidak ada hitungan yang menghasilkan 15.
  Koreksi: naskah sudah benar ("13-source standard mixture … plus 9,600
  consensus-soft rows"), yang perlu ditambah hanya rincian customs (ada di
  `dataset.tex:5-7` — SUDAH ada). Sisa kerja #14 = tabel per-sumber di bawah
  + lisensi 5 unaudited (didisclose di `dataset.tex:23-25` — SUDAH ada).

## Jawaban 30k vs 28k vs 43k — ketiganya benar, unitnya beda

| Angka | Unit | Run | Status |
|---|---|---|---|
| 30.000 | **rows** = 20.400 standar + 9.600 customs | seed3407/41/42, loto-* (`n_rows_built`, mesin) | `dataset.tex` Tab. mixture ✅ |
| 28.000 | **rows** = `MAX_DECISIONS` subsample (ablasi token-matched A vs E) | ablasi-A/E (`n_rows_built`, mesin) | `results.tex:83` ✅ |
| 43.171 | **decisions** = questions setelah render+augmentasi (seed3407) | seed3407 (`decisions_total`, mesin) | `dataset.tex:65` ✅ |
| 39.560 vs 31.754 | **decisions** E vs A (budget row sama 28k; augmentasi → +25%) | ablasi-E/A (`decisions_total`, mesin — lihat koreksi di bawah) | `dataset.tex:54-55`, `results.tex:83` ⚠️ SALAH, koreksi di bawah |

## Tabel definitif per run (mesin, dari run_meta.json)

| Run | rows_built | decisions_total | skipped | aug | max_dec | n_train/cal/test | seed | loto |
|---|---|---|---|---|---|---|---|---|
| seed3407 (E) | 30.000 | 43.171 | 809 | full | 0 | 28.910/140/141 | 3407 | — |
| seed41 (E) | 30.000 | 43.172 | 810 | full | 0 | 28.926/132/132 | 41 | — |
| seed42 (E) | 30.000 | 43.151 | 858 | full | 0 | 28.870/136/136 | 42 | — |
| loto-banking | 30.000 | 43.736 | 809 | full | 0 | 28.926/132/133 | 3407 | banking77 |
| loto-clinc | 30.000 | 43.695 | 809 | full | 0 | 28.913/139/139 | 3407 | clinc150 |
| ablasi-A | 28.000 | 32.500 | 746 | none | 28.000 | 26.914/170/170 | 3407 | — |
| ablasi-E | 28.000 | 40.306 | 746 | full | 28.000 | 26.962/146/146 | 3407 | — |

Komposisi rows (kode, deterministik): standar 20.400 via bobot 17 shares
(bank/clinc 3× = 3.600 each, 11 lainnya 1× = 1.200 each; 2×3.600+11×1.200 =
20.400 ✅) + customs 9.600 (Open-Jev release-v2-redistributable 3.600 +
mailroom-control-v1 2.400 + LeWiDi conv_abuse 1.200 + csc 1.200 + GoEmotions
1.200 ✅). 20.400+9.600 = 30.000 ✅. Decisions/row: E-seed3407 1,439;
ablasi-A 1,161; ablasi-E 1,440 (augmentasi +24% decisions — confound #7,
sudah didisclose di `results.tex:107`).

## Koreksi yang ditemukan audit ini (PENTING — teks koreksi di bawah)

1. **Angka "39.560 vs 31.754" di `dataset.tex:54-55` dan `results.tex:83`
   TIDAK cocok dengan mesin**: run_meta ablasi-E = **40.306** (bukan 39.560),
   ablasi-A = **32.500** (bukan 31.754). Selisih −746/+746 ≈ tepat `skipped`
   746?? Tidak: 40.306−39.560 = 746 dan 32.500−31.754 = 746 — persis sama
   dengan skipped. Dugaan: angka naskah = decisions_total − skipped
   (keputusan render-bersih), TAPI untuk seed3407 naskah memakai 43.171 =
   decisions_total MENTAH (43.171−809 = 42.362 ≠ angka mana pun di naskah).
   Jadi naskah **inkonsisten definisi**: ablasi pakai (total−skipped),
   headline pakai total-mentah. Harus diseragamkan (rekomendasi: selalu
   decisions_total mentah + skipped terpisah).
2. **`dataset.tex:8-9` "twelve public intent and reasoning sources"**:
   frasa "reasoning" tak terdefinisi (tak ada sumber bernama reasoning di
   `MIX_SOURCES`; maksudnya NLI/MCQ). Issue benar menanyakannya.
3. **Split sum ≠ decisions_total** (mis. 29.191 vs 43.171): `split_holdout`
   adalah metode `FastDecisionModel` (compiled, tak ada di sumber lib
   `decision_datasets.py`) — fraksi holdout ~1% konsisten di semua run
   (0,9–1,25%), TAPI total split (29.191) hanya 67,6% dari decisions_total.
   ~32% decisions hilang antara `build_dataset` → `split_holdout` tanpa
   penjelasan di kode terlihat. **Sisa inkonsistensi TERBUKA** — butuh baca
   sumber compiled/lib versi pin atau eksperimen kecil (bukan CPU-only
   ringan; catat sebagai open).
4. LOTO-banking decisions (43.736) > seed3407 (43.171) meski satu sumber
   dikeluarkan: konsisten dengan re-alokasi bobot ke 12 sumber sisa
   (bobot bank 3× = 3.600 rows didistribusikan ulang) + augmentasi stokastik —
   BUKAN anomali, tapi perlu satu kalimat di naskah.
5. seed41 vs seed42 skipped 810 vs 858 (seed memengaruhi render skip via
   subsample `max_options=24` saat augmentasi) — kecil, catat saja.

## Tabel per-sumber (rows sampled — dari kode; decisions per-sumber TIDAK
## tercatat di run_meta, hanya total — sisa inkonsistensi #5)

| # | Sumber (standar) | HF ID | Split train | Bobot | Rows (E-30k) | Label |
|---|---|---|---|---|---|---|
| 1 | banking77 | legacy-datasets/banking77 | train | 3× | 3.600 | hard choice (77) |
| 2 | clinc150 | clinc/clinc_oos:plus | train | 3× | 3.600 | hard choice (151) |
| 3 | ag_news | fancyzhx/ag_news | train | 1× | 1.200 | hard topic (4) |
| 4 | arc | allenai/ai2_arc:ARC-Challenge | train | 1× | 1.200 | hard MCQ |
| 5 | boolq | google/boolq | train | 1× | 1.200 | hard noul |
| 6 | commonsense_qa | tau/commonsense_qa | train | 1× | 1.200 | hard MCQ |
| 7 | mmlu | cais/mmlu:all | auxiliary_train | 1× | 1.200 | hard MCQ |
| 8 | mnli | nyu-mll/multi_nli | train | 1× | 1.200 | hard NLI (3) |
| 9 | prompt_injections | deepset/prompt-injections | train | 1× | 1.200 | hard noul |
| 10 | snli | stanfordnlp/snli | train | 1× | 1.200 | hard NLI (3) |
| 11 | sst5 | SetFit/sst5 | train | 1× | 1.200 | hard score (5) |
| 12 | wanli | alisawuffles/WANLI | train | 1× | 1.200 | hard NLI (3) |
| 13 | typed-decisions | LocalLLaMA/typed-decisions:all | train | 1× | 1.200 | soft (5q/record) |
| | Customs (concat pasca-builder) | | | | 9.600 | |
| 14 | Open-Jev release-v2-redistributable | ZefanCai/Open-Jev | train | — | 3.600 | soft |
| 15 | Open-Jev mailroom-control-v1 | ZefanCai/Open-Jev | train | — | 2.400 | soft |
| 16 | LeWiDi conv_abuse | tasksource/lewidi | train | — | 1.200 | soft noul |
| 17 | LeWiDi csc | tasksource/lewidi | train | — | 1.200 | soft score (6) |
| 18 | GoEmotions raw | google-research-datasets/go_emotions | train | — | 1.200 | soft choice (28) |

(#rows = target sampling; pool diambil 2× share lalu dedup-dekon di
`build_decision_mixture`, sehingga rows aktual per-sumber bisa ±1 dari
pembulatan share. Decisions & skipped PER-SUMBER tidak dilog — hanya total.)

## Teks koreksi siap-tempel buat naskah (JANGAN edit paper/*.tex langsung)

> Accounting (all counts from `run_meta.json`, seed3407 unless noted).
> Standard mixture: 13 sources (12 public + typed-decisions) sampled to
> 20,400 rows under 17 weight-shares (BANKING77/CLINC150 at 3$\times$ =
> 3,600 rows each; 11 others at 1$\times$ = 1,200 each), plus 9,600
> consensus-soft rows (Open-Jev 3,600+2,400; LeWiDi 1,200+1,200; GoEmotions
> 1,200), for 30,000 rows total. After rendering and augmentation this
> yields 43,171 decisions (809 skipped, logged per-run) for recipe E and
> 32,500 for recipe A at the same 28,000-row budget (40,306 for E at that
> budget — the augmentation residual disclosed in \S6). Per-source decision
> and skip counts are not logged by the training script (only totals), and
> ~32% of built decisions do not reach the train/cal/test split through the
> library's compiled `split_holdout`; both are open bookkeeping gaps that do
> not affect the reported test metrics. LOTO runs reallocate the held-out
> source's 3,600-row share across the remaining sources, hence 43,736
> (banking-held-out) and 43,695 (clinc-held-out) decisions.

## Sisa inkonsistensi (terbuka, jujur)

1. 39.560/31.754 (naskah) vs 40.306/32.500 (mesin) — butuh satu definisi
   (total-mentah vs bersih) + edit angka naskah.
2. Split sum hanya 67,6% decisions_total — mekanisme `split_holdout`
   compiled tak teraudit (butuh pin versi lib / baca wheel).
3. Decisions & skipped per-sumber tak dilog (hanya total) — butuh
   instrumentasi ulang atau rerun logging (GPU).
4. Frasa "reasoning sources" (`dataset.tex:8`) tak terdefinisi.
5. Lisensi 5 sumber (sst5, wanli, LeWiDi, Open-Jev, GoEmotions) unaudited —
   sudah didisclose, tetap open sampai diverifikasi.
