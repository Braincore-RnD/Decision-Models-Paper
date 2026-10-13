---
title: Roast board — 22 issue critical + major dan rencana perbaikan
tags: [paper, review, roast, revision-plan]
---

Kumpulan 22 issue terbuka label `roast` + `severity: critical|major` dari
[issues repo](https://github.com/Braincore-RnD/Decision-Models-Paper/issues)
(semua oleh `algonacci`, 2026-10-13). Minor (#26–28) tidak dicakup.
Terverifikasi terhadap tree lokal sebelum ditulis.
Terkait: [[08-paper-outline]], [[RECOMMENDATION]], [[RUN_LOG]].

> [!quote] Syarat accept tiap issue dikutip dari bodi issue-nya; link per issue di heading.

## Critical (desk-reject level): #4–#12

### #4 — "Chance" baseline itu bukan chance
[#4](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/4):
uniform chance CLINC150 = 0.67%, BANKING77 = 1.3%, tapi paper menjual
untrained-head 17.2%/2.4% sebagai "chance". Itu zero-shot baseline gratisan (~25× chance), bukan chance.
- Rencana: ganti istilah → "untrained-head baseline" di `abstract.tex:10`, `introduction.tex:25`, `results.tex:8,32-33`, `experiment.tex:19`; tambah kolom uniform chance; tambah baseline cosine-similarity embedding (bge-small, CPU lokal) + lexical-prior-only head untuk LOTO; klaim transfer relatif ke baseline itu.

### #5 — Nol baseline yang dijalankan sendiri
[#5](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/5):
paper menyerang generative router lalu tidak menjalankan satu pun; 81.3/72.9 di bawah literatur encoder era 2019–2020 tanpa pembanding; "1.5–4 s" tanpa sitasi/pengukuran.
- Rencana (butuh GPU): 3 baseline di test 500/500/400 identik — (a) ModernBERT-base fine-tune, (b) LFM2.5-350M generatif JSON (ukur acc/ECE/latency/violation rate), (c) zero-shot embedding untuk LOTO. Tanpa ini klaim "viable" tak teruji.

### #6 — "Augmentation buys transfer" tidak pernah diuji
[#6](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/6):
LOTO hanya ada dengan `aug=full`; sel LOTO × canonical (A) kosong; `results.tex:222-223` mengakui atribusi mustahil tapi abstrak tetap klaim.
- Rencana (butuh GPU, mahal): desain 2×2 {A,E} × {in-dist, LOTO-bank, LOTO-clinc}, ≥3 seed/sel = 6 train Kaggle tambahan. Murah: hapus dari abstrak/kontribusi → turunkan jadi hipotesis (teks saja).

### #7 — Ablasi A vs E single-seed, efek tenggelam di seed-noise
[#7](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/7):
selisih A−E 7 poin = seed spread recipe E sendiri (83.8−76.8 = 7.0); plus confound +25% decisions & batch 4×8 vs 8×4; E menang di typed (59 vs 54) tak dibahas.
- Rencana: A ×3 seed + E ×2 seed tambahan (Kaggle); subsample E ke 31,754 decisions (flag di train script); McNemar/paired bootstrap di `score_preds.py`; bahas counter-evidence typed.

### #8 — "Calibration under shift" bandingkan dua model berbeda
[#8](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/8):
baris "E" vs baris "LOTO" = dua training run; ECE naik saat akurasi 80→43 adalah ekspektasi default; TS-transfer tak pernah diukur (butuh ECE raw vs TS-in-dist vs TS-oracle).
- Rencana (bisa offline dari dumps): redesain Tab. rq3 satu model (LOTO-banking: clinc in-dist ECE 0.176 vs banking shift 0.366 — data sudah ada); refit TS-oracle di `items_pred.json`; tambah reliability diagram + risk–coverage + AURC ternormalisasi.

### #9 — CI di paper tidak ada di log (semua ci95 = NaN)
[#9](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/9).
Terverifikasi lokal:
`docs/eval/kaggle/seed3407/eval_results.json` → `"ci95": {"typed_decisions": [NaN, NaN], "banking77": [NaN, NaN], "clinc150": [NaN, NaN]}`.
Plus: 44 (abstrak) vs 43.4 (log) vs 43 (Tab. rq3) untuk model×bench yang sama; bootstrap 100 resamples terlalu sedikit; field `seed: 3407` tercatat di run seed41/seed42.
- Rencana (CPU lokal, tanpa GPU): regenerate semua CI dari per-item dumps (`items_pred.json` ada di `docs/eval/colab/loto-banking/`, `ablasi-full/`) ≥1000 resamples; satu angka per model×bench; dokumentasikan eval-seed vs train-seed.

### #10 — Dekontaminasi 13-gram ≈ tidak ada dekontaminasi
[#10](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/10):
utterance rata-rata ~10 kata < 13 token → shingle tak pernah menembak; ke-12 sumber tak disebut nama; risiko bocor LOTO via mirror HF.
- Rencana: sebut 12 sumber (nama, versi HF, split, baris); redo exact-match ternormalisasi + 5–8gram + embedding near-dup (cosine > 0.9); audit LOTO eksplisit; laporkan baris dibuang/sumber. Risiko: bila banyak bocor → retrain (mahal, Fase 4).

### #11 — Integritas referensi
[#11](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/11).
Terverifikasi lokal: `related_works.tex:10` typo "tributeevaluate"; `unsloth-decision` di bib tapi tak pernah di-cite.
- Rencana (teks saja): verifikasi tiap entri bib (ID resolve, author lengkap — hapus `author={et al.}`); `@misc` + `howpublished` + `note={Accessed: ...}` untuk blog/HF; cite `unsloth-decision` di §4.5; fix typo.

### #12 — "License text is still unread" di naskah + reproducibility kosong
[#12](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/12).
Terverifikasi lokal: `results.tex:229` masih memuat kalimat tersebut; checkpoint di path Kaggle privat akun non-author; nol URL repo/HF/DOI di paper; bench `typed-decisions` (400 item) asal-usul tak dijelaskan.
- Rencana (manusiawi, tanpa GPU): baca lisensi LFM1.0 **sekarang**; rilis adapter + head + dumps + eval script + typed bench di HF/Zenodo (DOI); cantumkan URL; hapus kalimat.

## Major: #13–#25

### #13 — Test subset 500, unit hitung tak konsisten
[#13](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/13):
cara sampling 500 tak dijelaskan; slice OOS CLINC150 (paling relevan untuk ESCALATE) diabaikan; caption "500/500/400" urutan terbalik; typed per-record (400) vs per-question (2000) di kalibrasi.
- Rencana (eval-only, GPU murah): evaluasi full test resmi (3,080/5,500); definisikan unit per metrik; tambah slice OOS + metrik abstain.

### #14 — Akuntansi data tak konsisten (13 vs 15 sumber, 30k vs 28k vs 43k)
[#14](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/14).
- Rencana (audit teks dari train logs): satu tabel — nama sumber, lisensi, #rows, #decisions, #skipped, label type; semua angka menjumlah persis di tiap bagian.

### #15 — Novelty tipis: head "attached by default", klaim hybrid tak diuji
[#15](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/15):
`methodology.tex:33-35` mengakui arsitektur bawaan library; nol dense transformer pembanding; judul bertanya "what makes transfer" dengan satu model satu ukuran.
- Rencana (GPU): recipe identik di SmolLM2-360M/Qwen2.5-0.5B + ModernBERT-base; atau hapus klaim hybrid + reposisi jujur sebagai empirical study + sesuaikan judul.

### #16 — "Capacity frontier" tanpa eksperimen
[#16](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/16):
gap vs angka vendor Qwen (protokol beda, diakui paper sendiri) disimpulkan "architectural capacity, not recipe".
- Rencana (GPU): recipe identik di backbone lebih besar (LFM2-700M/1.2B) atau turunkan jadi hipotesis.

### #17 — Tabel external context kontradiksi kalimatnya sendiri
[#17](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/17):
"no cross-row ranking" lalu "recipe dominates size" (accuracy vs macro-F1 dicampur); "the board" tak didefinisikan; single-seed best run disejajarkan dengan mean.
- Rencana (teks): hapus "recipe dominates size" (atau rerun Laya di protokol kita); **jangan sitasi "the board"** — itu reviewer simulasi internal, bukan sumber; keluarkan single-seed dari tabel.

### #18 — Metode tak terreproduksi: λ, param count, lexical prior, dual temperature
[#18](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/18):
λ_Brier/λ_ord tanpa nilai; lexical prior tak didefinisikan (padahal sumber 17% "chance" — terkait #4); param count trainable tak dilaporkan; "dual" padahal `fitted_types` ada tiga; biaya prefill vs |options| tak dibahas.
- Rencana (teks + 1 perintah): persamaan lengkap head; param count dari checkpoint (`sum(p.numel())` per komponen); definisi TS; ubah klaim O(1) → "O(1) decoding steps, O(L) prefill, L tumbuh dengan |options|".

### #19 — Calibration set ~136 item untuk multi-temperature + cross-fitting
[#19](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/19):
~68 item/fold untuk 3 tipe; ECE 10-bin ≈ 13 item/bin (bias & varians besar); angka cal in-dist tanpa CI.
- Rencana: cal set ≥1k (cek apakah logits tersimpan di dumps → refit offline mungkin); ECE + CI bootstrap, equal-mass/debiased ECE; atau Brier/NLL sebagai metrik utama.

### #20 — Selection berbagi data dengan kalibrasi; seed3407 hilang dari tabel
[#20](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/20):
best-checkpoint via eval-loss di split yang sama dengan fit TS (bias optimis); Tab. overfit 6 run padahal 7; "epoch 2 always wins" ×3 (cosine 3-epoch dipotong = LR belum turun penuh).
- Rencana: pisahkan val/cal/test eksplisit; tampilkan seed3407 atau jelaskan; tambah training curve (figure, terkait #25).

### #21 — Latency tak dibandingkan; "no KV-cache growth" tak diukur; Laya 10–100× lebih cepat
[#21](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/21):
56–249 ms satu kalimat tanpa tabel; CLINC ~221 ms vs typed ~56 ms = biaya skala dengan |options| (prefill), bukan O(1).
- Rencana (eval-only): tabel latency + memori + throughput (batch 1 & besar) untuk model kita vs generatif vs encoder; latency sebagai fungsi |options|.

### #22 — Klaim bersandar eksperimen tak dilaporkan (36→83%, doubling typed share)
[#22](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/22):
run 2048-ctx dan mixture-ratio tak ada di tabel/run-ID manapun — anekdot internal sebagai temuan #2.
- Rencana: masukkan ke tabel (run 2048 vs 4096 × {A,E} + rows terpotong/skip; mixture-ratio + seed) atau hapus.

### #23 — Paraphrase grid inkonsisten (3 vs 2 varian, typed tak diparafrase, seed runs kosong)
[#23](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/23):
`experiment.tex:9` "3-variant" vs Algorithm 1 "2 variants"; typed range nol; tiga run utama `"paraphrase": {}`; "schema semantics" dari 3 varian template.
- Rencana (eval-only murah): samakan jumlah; grid untuk semua run×bench; contoh prompt di appendix; uji semantik sungguhan via label-name shuffle/sinonim (terkait klaim semantik).

### #24 — Overclaiming ("honest numbers", "none touch headline findings", "calibrated")
[#24](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/24):
`results.tex:230-231` mengklaim limitations tak menyentuh temuan padahal menghantam #2 langsung; "calibrated" vs ECE 0.121–0.366; "load-bearing" bukan prosa ilmiah.
- Rencana (teks): hapus self-assessment; petakan tiap limitation → temuan yang terdampak.

### #25 — Nol figure; Algorithm 1 bukan algoritma; tabel redundan
[#25](https://github.com/Braincore-RnD/Decision-Models-Paper/issues/25):
tanpa reliability diagram untuk paper kalibrasi; tanpa diagram arsitektur; Algorithm 1 = checklist; Tab. loto ≈ teks ≈ Tab. para.
- Rencana: 4 figure (arsitektur, reliability in-dist vs shift, risk–coverage, loss curve — data dumps sudah ada); gabung Tab. loto→para; Algorithm 1 jadi figure pipeline atau hapus.

## Roadmap urutan kerja

- **Fase 0 — teks saja, tanpa GPU**: #11 (bib+typo), #24 (nada), #4 (rename chance + uniform), #17 (hapus ranking), #14 (audit tabel), #18 (persamaan + param count), #20 (seed3407 + split), #22 (tabel atau hapus).
- **Fase 1 — CPU lokal dari dumps**: #9 (CI ≥1000 resamples), #8 (RQ3 satu model + TS-oracle), #25 (4 figure), #19 (refit TS bila logits tersimpan).
- **Fase 2 — eval-only GPU murah (Colab)**: #13 (full test + OOS), #23 (paraphrase semua run + label-shuffle), #21 (tabel latency/memori), #4/#5c (baseline embedding bila belum di Fase 1).
- **Fase 3 — train GPU (Kaggle, ~10–15 jam dari kuota 30 jam)**: #5a/b (encoder + generatif), #6 (sel A-LOTO, atau pangkas klaim), #7 (seed tambahan + subsample), #15/#16 (backbone pembanding, atau pangkas klaim).
- **Fase 4 — rilis & manusiawi**: #12 (baca LFM1.0 + HF/Zenodo DOI + URL), #10 (sumber + redo dekon; retrain hanya bila bocor besar).

## Eksekusi Fase 0 — PR #29 (`feat/roast-fase0`, 2026-10-13, teks saja, 0 GPU)

Build: `main.pdf` 6 halaman, 0 Overfull, 0 Error, 0 undefined
(`\usepackage{url}` ditambah — `\url` di bib error dari sononya).
Status per issue — DONE / PARTIAL / OPEN:

- #4 DONE (teks): "chance" → untrained-head 30.6/2.4/17.2 (stub log, deterministik antar-run) + uniform 1.3/0.67 di 6 titik + Tab. loto. Baseline embedding → Fase 2.
- #5 OPEN (Fase 3). #7 OPEN (Fase 3).
- #6 PARTIAL: "buys" → "coincides with" (abstrak/intro/results) karena sel A-transfer memang tak ada; 2×2 → Fase 3.
- #8 PARTIAL: kalimat same-model ditambah (LOTO-banking: CLINC in-dist ECE 0.176 → BANK shift 0.366, data sudah di Tab. rq3); TS-oracle + diagram → Fase 1.
- #9 PARTIAL: LOTO diseragamkan 43.4; "100 resamples" (tak sesuai log) → N_BOOT 30/0 sesuai RUN_LOG §7; regen ≥1000 → Fase 1. Field `seed: 3407` di run seed41/42 = eval-seed default, didokumentasikan di sini (bukan di paper).
- #10 OPEN (Fase 4). #12 PARTIAL: URL repo publik masuk §5; lisensi + rilis → Fase 4.
- #11 DONE: 10/10 ID arXiv resolve via API; judul + penulis dibetulkan (2 `author={et al.}` → nama asli); `liquid-d1` → `@misc`; `+Accessed: 2026-10-13`; typo fixed; `unsloth-decision` di-cite §4.5. Koreksi terhadap issue: tuduhan `2609.38850` SALAH (valid, cocok); ρ=0.87, TypedBench grid/noise-floor, JevBench math-gap terkonfirmasi dari abstrak. Sitasi `bottled2026` DICABUT (isi aslinya agent-bottling, bukan finetune-360M).
- #13 PARTIAL: caption "(typed 400; BANK/CLINC 500 each)"; full test + OOS → Fase 2.
- #14 DONE (teks): Tab. mixture menjumlah 20.400+9.600=30.000 eksak; 13 = 12 publik + typed; baris-bobot bogus jadi footnote; "~800/43.000" (tak terverifikasi lokal) → "43.171 decisions (seed3407, RUN_LOG §7)" + skip counts di `run_meta.json`. Tabel lisensi per-sumber tetap OPEN.
- #15/#16 OPEN. #19 OPEN. #21 OPEN.
- #17 DONE: "recipe dominates size" + "the board" + baris single-seed dibuang; paragraf jadi deskriptif murni.
- #18 DONE (teks): "Dual" → 1 temperatur per tipe (`fitted_types: [0,1,2]` di semua stub); λ = library default (skrip tak override `DecisionTrainer` — terverifikasi); lexical prior didefinisikan + dikaitkan ke #4; param = konfigurasi (total eksak tak tercatat di `run_meta.json` — dibuka, bukan dikarang). Klaim O(1)→prefill → Fase 2.
- #20 DONE (teks): sharing cal/selection dibuka (`eval_dataset=cal_items` di kode) + janji split validasi; seed3407 dijelaskan di caption; "epoch 2 always wins" tinggal 2 sebutan (protokol + tabel). Curve → Fase 1 (#25).
- #22 DONE (teks): tuduhan "tanpa run ID" terbantahkan parsial — Tab. precursors v5/v6 + v2/v3 + kernel slug (`RUN_LOG.md` §1–§2, `docs/07-mixture-compositions.md`); v5 ctx 2048 + skip 27% vs v6 skip 5% konsisten dengan cerita truncasi. Klaim mixture-ratio v3 ("banking +2, clinc −7") memang tercatat di RUN_LOG §2.
- #23 PARTIAL: ambiguitas 2-vs-3 selesai — kode (`_INTENT_ALTS` 3 entri, varian 0 = kanonis) → teks "canonical + 2 variants"; typed range-nol + grid ulang → Fase 2.
- #24 DONE: "honest/reproduced/none-touch/calibrated/load-bearing" dibuang; limitations dipetakan ke 3 temuan.
- #25 OPEN (Fase 1; Algorithm 1 dipertahankan sementara).

## Eksekusi Fase 1 lokal — PR #30 (`feat/roast-fase1`, CPU saja)

Dumps `items_pred.json` ternyata menyimpan full prob vector + gold → 80% Fase 1
jalan lokal. Berat terukur: bootstrap 2000 × 3 bench ≈ 70 detik/file;
figure <3 detik (`uv run --with matplotlib`, tanpa install).

- #9 DONE (parsial, tanpa GPU): `score_preds.py --boot 2000` dari dumps →
  `docs/eval/colab/{loto-banking,ablasi-full}/metrics_ci.json` (baru, file lama
  tak disentuh). LOTO-banking 43.4 [39.4–47.6]; E-banking 79.6 [76–83].
  A dan LOTO-clinc tanpa dumps → point estimates + disclosure di caption.
- #8 DONE (parsial): TS-oracle refit per-tipe dari prob vector (eksak via
  shift-invariance softmax) → Tab. `tab:ts` baru: LOTO-shift 0.366 → TS-id
  0.220 → oracle 0.104. Klaim "does not transfer" direvisi jujur jadi
  "transfers partially" (abstrak/intro/results/discussion). TS-oracle penuh +
  AURC → butuh cal dumps (Colab).
- #25 PARTIAL: `fig_reliability.pdf` + `fig_riskcover.pdf` (matplotlib lokal,
  E-in-dist vs LOTO-shift) + diagram tikz `figures/arch.tex`. Naskah 6→7
  halaman. Loss-curve figure → butuh train console (Colab). Tab. loto → para
  belum digabung; Algorithm 1 dipertahankan.
- #19 OPEN: butuh `cal_items.pt` (hanya di kernel output) → Colab.

## Eksekusi Colab (masih PR #30, `feat/roast-fase1`)

Kaggle CLI 2.x butuh OAuth (`kaggle auth login`, user menjalankannya);
token env lama 401 di kedua format. Jejak: tidak ada secret di repo.

- Kernel outputs diunduh lokal (`kaggle kernels output` ×7): kernel **stdout
  log** (~300KB) ternyata memuat full train-loss + eval-loss per epoch, plus
  `run_meta.json` + `cal/test_items.pt` per run. Yang di-commit (kecil,
  traceable): `docs/eval/kaggle/*/loss_series.json` + `run_meta.json`;
  `.pt`/log penuh tetap di `/tmp` (tak masuk repo).
- #20 CLOSED total + koreksi jujur: klaim "epoch 2 always wins" **salah
  untuk 1 dari 7 run** — seed3407 best = ep1 (0.6495 < 0.6965 < 0.8421).
  Tab. overfit sekarang 7 baris × 3 epoch + Figure loss-curve;
  teks: "best at epoch 1 or 2, never 3".
- #19 PARTIAL (butuh 1 sesi T4, lalu stop): `final/` seed3407 (76MB) +
  `cal_items.pt` di-upload; `_decision_logits` → `cal_logits.json`
  (192 questions) → `cal_metrics.json`: raw ECE **0.119 [0.084–0.191]**,
  equal-mass 0.122, cross-fitted 0.081 (logged), in-sample TS 0.067
  (T 6–13, overconfident). CI ±5pt = bukti kuantitatif klaim #19;
  kalimatnya masuk §4. Cal ≥1k tetap butuh perubahan protokol (future work).
- Colab T4 sempat ke-block orphan assignment (~30 mnt) lalu Sessions READY;
  sesi di-stop setelah dump (no burn). `uv run --with matplotlib` untuk
  figure (tanpa install).

## Eksekusi Fase 2 (masih PR #30) — full-test E di Kaggle, probe di Colab

Kuota Kaggle: GPU sisa 2.39/30h (refresh 17 Okt) → 1 job国内 hati-hati.
TPU 20h tak terpakai (Unsloth butuh CUDA).

- #13 PARTIAL: kernel `amalanfadilgaib/lfm-decs1-v7-eval-e-full`
  (`eval_agg.py`, batched BS=4, N_BOOT=0) → `docs/eval/kaggle/eval-E-full/agg.json`:
  E full-test typed **58.8** [53.9–63.5] / BANK **80.6** [79.1–81.9] (3.080) /
  CLINC **69.4** [68.2–70.6] (5.500), in-scope 79.6, **OOS-recall 23.5%**
  [21.0–26.2] (1.000, Wilson CI). Full ≈ subset (79.6/69.8) → ranking subset
  transfer. Seeds + LOTO full-test tak muat kuota → subset + disclosure.
- #23 PARTIAL (semantik, E, Colab `eval_probe.py`): label-shuffle
  (nama diacak, deskripsi tetap) BANK 79.6→**5.6%**, CLINC 69.8→**3.0%** —
  sinyal ada di semantik nama label. Para-grid ulang seeds tertunda
  (bobot seed41/42 tak terunduh saat auth hidup).
- #21 PARTIAL (latency per-row T4 4-bit, E): typed 139–148ms/row,
  BANK77 245–288ms (1.2GB), CLINC150 515–518ms (2.3GB); top8/24 134/116ms
  tapi akurasi hancur (9.6/28.8%) karena gold terpotong. Klaim lama 56–249ms
  direvisi di naskah. Batched ~2× lebih cepat (127/221ms) — deployment =
  angka per-row. Perbandingan vs generatif/encoder → Fase 3 (butuh modelnya).
- Infra: VM CLI mati ~6× di umur 30–40 mnt → pola kerja = job <20 mnt +
  `RESUME=1`/save-per-chunk + download langsung (`eval_full.py`,
  `eval_probe.py` MODE/KS/BENCH filter). Kernel Kaggle 2× ERROR sebelum jalan
  (FINAL env, lalu unsloth tak terinstal — keduanya di-fix di skrip).
  Kernel `ablasi-none-full` + semua kernel eval 403 permanen (tak bisa
  diunduh); A tetap tanpa bobot → subset + disclosure.
