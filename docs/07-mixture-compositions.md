---
title: "Komposisi dataset per versi v1–v6 (catatan lengkap mixture)"
date: 2026-10-08
tags:
  - decision-models
  - dataset
  - mixture
  - run-log
aliases:
  - Mixture Compositions
---

# Komposisi dataset per versi (v1–v6)

> [!abstract] Definisi
> Catatan presisi: sumber, bobot, jumlah baris, augmentasi, dan dekontaminasi
> tiap run. Semua run memakai `seed=3407`, dekontaminasi shingle-13 terhadap
> test typed-decisions/BANKING77/CLINC150, dan holdout `split_holdout` 10%
> (maks 400). Induk run: `../from_scratch/README.md`.

---

## v1 — typed saja (kernel `…-decs1-unsloth`)

- Sumber: `LocalLLaMA/typed-decisions:all`, split train, 1.200 baris / 6.000 decisions.
- Augmentasi: tidak. Split: 1120 latih / 80 holdout. Epoch: 2.

## v2 — mixture equal (kernel `…-decs1-unsloth`, direrun dengan script mixture)

- `build_decision_mixture`, 13 sumber bobot equal (12 publik + typed),
  N=12.000 baris → 20.252 decisions (skipped ~0), augmentasi ON.
- Split: 11.765 / 235. Epoch: 3 (spike eval_loss di ep-3).

## v3 — typed 2× (kernel `…-decs1-v3`)

- Sama dengan v2 kecuali `typed_decisions` bobot 2.0. N=12.000, augmentasi ON.
- Epoch: 2. (Jumlah decisions tidak tercatat di log; estimasi ~20k.)

## v4 — 30k no-augment + lunak (kernel `…-decs1-v4` v2)

- Standar: 13 sumber bobot equal, N=20.000, augmentasi OFF.
- Customs (tanpa augmentasi, converter lokal, didekontaminasi):
  Open-Jev `release-v2` 5.750 (4300+1450 digabung pasca-gagal `customer-control-v1`),
  LeWiDi `conv_abuse`/`csc` 1.450 × 2, GoEmotions 1.450 → 10.100 baris.
- Total: 36.252 decisions (skipped 5.809 = 16%), split 23.979 / 312. Epoch: 3.

## v5 — rebalance intent (kernel `…-decs1-v5`)

- Standar: N=20.400, bobot banking77 3× + clinc150 3×, sisanya 1×
  (typed kembali 1×). Augmentasi OFF.
- Customs: Open-Jev `release-v2` 3.600 + `mailroom-control-v1` 2.400,
  LeWiDi 1.200 × 2, GoEmotions 1.200 → 9.600 baris.
- Total: 34.800 decisions (skipped 9.442 = 27%). Epoch: 3.

## v6 — resep v5 + augmentasi ON (kernel `…-decs1-v6`)

- Identik v5 kecuali `AUGMENT=1` (berlaku untuk mixture standar; customs
  canonical tetap karena di-concat pasca-builder).
- Total: 43.171 decisions (skipped 2.242 = 5%). Epoch: 3.

## Pelajaran komposisi

1. Augmentasi ON/OFF adalah variabel dominan transfer (v3/v6 vs v1/v4/v5).
2. Bobot intent 5%→32% hanya menggeser ±10 poin (v4→v5).
3. Skip rate naik dengan customs (0%→16%→27%) lalu turun dengan augmentasi (5%) —
   mekanisme plausible: subsample `max_options=24` saat augmentasi memangkas skema
   lebar hingga muat budget; sisanya baris yang gagal validasi format
   (`conv_abuse` noul-criteria; lihat §3 note 04).
4. `customer-control-v1` bukan subset valid (hanya group_id); daftar subset sah
   Open-Jev ada di komentar `code/unsloth_kaggle_dec.py`.
