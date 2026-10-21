"""End-to-End Google Colab Script: Fine-Tuning Encoder / BERT / SBERT Models
menjadi System One Decision Models, Kalibrasi Pasca-Latih, Evaluasi 3 Benchmark
(BANKING77, CLINC150, Typed Decisions), dan Otomatis Push ke Hugging Face Hub.

Kompatibel dengan GPU T4 (Gratis di Colab).
Dapat dijalankan langsung via CLI atau disalin ke cell Google Colab:
  !pip install -q transformers datasets accelerate huggingface_hub
  python colab_train_eval_encoder_dec.py --model_id answerdotai/ModernBERT-base
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)


# ==============================================================================
# 1. Definisi Arsitektur Option-Marker Decision Model (Laya-Style)
# ==============================================================================
class OptionMarkerDecisionModel(nn.Module):
    """Backbone Bidirectional Encoder + Contextual [MASK] Head.
    Mengevaluasi seluruh opsi keputusan dalam satu forward pass (O(1)).
    """

    def __init__(self, backbone: PreTrainedModel, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )
        self.log_temp = nn.Parameter(torch.zeros(()))  # Inisialisasi T = 1.0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.05, max=10.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor,
        mask_valid: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # (B, L, D)

        B, max_opts = mask_positions.shape
        D = H.size(-1)

        # Ambil representasi vektor pada posisi token [MASK]
        idx_exp = mask_positions.unsqueeze(-1).expand(B, max_opts, D)
        mask_reps = torch.gather(H, dim=1, index=idx_exp)  # (B, max_opts, D)

        # Proyeksikan ke logit skalar per opsi
        logits = self.head(mask_reps).squeeze(-1)  # (B, max_opts)
        logits = logits.masked_fill(~mask_valid, -1e4)

        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(scaled_logits, labels, label_smoothing=label_smoothing)

        return {
            "logits": logits,
            "scaled_logits": scaled_logits,
            "probabilities": probs,
            "loss": loss,
        }


# ==============================================================================
# 2. Utilitas Pemrosesan Teks & Formatting [MASK]
# ==============================================================================
def format_decision_row(
    tokenizer: PreTrainedTokenizer,
    state: str,
    question: str,
    criteria: Dict[str, str],
    gold_option: str | None = None,
    max_len: int = 512,
) -> Dict[str, Any] | None:
    """Menyisipkan token [MASK] sebelum deskripsi setiap opsi kriteria."""
    mask_token = tokenizer.mask_token or "[MASK]"
    lines = [f"State: {state.strip()}", f"Question: {question.strip()}", "Options:"]

    option_keys = list(criteria.keys())
    if not option_keys:
        return None

    for opt_key, opt_desc in criteria.items():
        desc_clean = str(opt_desc).strip() if opt_desc else opt_key
        lines.append(f"{mask_token} {opt_key}: {desc_clean}")

    full_text = "\n".join(lines)
    enc = tokenizer(full_text, max_length=max_len, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attn_mask = enc["attention_mask"][0]

    mask_id = tokenizer.mask_token_id
    mask_indices = (input_ids == mask_id).nonzero(as_tuple=True)[0].tolist()

    # Validasi: Jika terpotong max_length sehingga ada opsi yang hilang, lewati baris
    if len(mask_indices) != len(option_keys):
        return None

    label_idx = option_keys.index(gold_option) if gold_option in option_keys else -1

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "mask_indices": mask_indices,
        "options": option_keys,
        "label": label_idx,
    }


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_seq_len = max(len(b["input_ids"]) for b in batch)
    max_opts = max(len(b["mask_indices"]) for b in batch)
    B = len(batch)

    input_ids = torch.full((B, max_seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_seq_len), dtype=torch.long)
    mask_positions = torch.zeros((B, max_opts), dtype=torch.long)
    mask_valid = torch.zeros((B, max_opts), dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]

        n_o = len(b["mask_indices"])
        mask_positions[i, :n_o] = torch.tensor(b["mask_indices"], dtype=torch.long)
        mask_valid[i, :n_o] = True
        labels[i] = b["label"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_positions": mask_positions,
        "mask_valid": mask_valid,
        "labels": labels,
    }


class ListDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


# ==============================================================================
# 3. Pemuatan & Standardisasi Benchmark Datasets
# ==============================================================================
def load_and_prep_datasets(
    tokenizer: PreTrainedTokenizer,
    seed: int = 3407,
    max_train_samples: int = 10000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Memuat LocalLLaMA/typed-decisions untuk training & validation split."""
    from datasets import load_dataset

    print("Mengunduh dataset pelatihan: LocalLLaMA/typed-decisions...", flush=True)
    raw_ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")

    processed = []
    rng = random.Random(seed)
    indices = list(range(len(raw_ds)))
    rng.shuffle(indices)

    for idx in indices:
        if len(processed) >= max_train_samples:
            break
        row = raw_ds[idx]
        try:
            questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
            golds = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
            state = row["state"]

            # Pilih pertanyaan bertipe choice jika ada
            for qid, qdata in questions.items():
                if qdata.get("type") == "choice" and "criteria" in qdata:
                    gold_entry = golds.get(qid)
                    if isinstance(gold_entry, dict):
                        gold_ans = gold_entry.get("label") or gold_entry.get("choice") or gold_entry.get("answer")
                    else:
                        gold_ans = gold_entry
                    if not gold_ans:
                        continue
                    item = format_decision_row(
                        tokenizer,
                        state=state,
                        question=qdata.get("instructions", "Select the best option:"),
                        criteria=qdata["criteria"],
                        gold_option=str(gold_ans),
                    )
                    if item and item["label"] != -1:
                        processed.append(item)
                        break
        except Exception:
            continue

    print(f"Total baris valid yang diproses untuk training/val: {len(processed)}", flush=True)
    split_idx = int(0.9 * len(processed))
    return processed[:split_idx], processed[split_idx:]


def load_benchmark_test_set(
    bench_name: str,
    tokenizer: PreTrainedTokenizer,
    limit: int = 500,
    seed: int = 3407,
) -> List[Dict[str, Any]]:
    """Memuat benchmark canonical: 'typed_decisions', 'banking77', atau 'clinc150'."""
    from datasets import load_dataset

    items = []
    rng = random.Random(seed)

    if bench_name == "typed_decisions":
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            if len(items) >= limit:
                break
            row = ds[idx]
            try:
                questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
                golds = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
                for qid, qdata in questions.items():
                    if qdata.get("type") == "choice" and "criteria" in qdata:
                        gold_entry = golds.get(qid)
                        if isinstance(gold_entry, dict):
                            gold = gold_entry.get("label") or gold_entry.get("choice") or gold_entry.get("answer")
                        else:
                            gold = gold_entry
                        if not gold:
                            continue
                        fmt = format_decision_row(
                            tokenizer,
                            state=row["state"],
                            question=qdata.get("instructions", "Choose an option:"),
                            criteria=qdata["criteria"],
                            gold_option=str(gold),
                        )
                        if fmt and fmt["label"] != -1:
                            items.append(fmt)
                            break
            except Exception:
                continue

    elif bench_name == "banking77":
        ds = load_dataset("legacy-datasets/banking77", split="test")
        label_names = ds.features["label"].names
        criteria = {lbl: lbl.replace("_", " ") for lbl in label_names}
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            if len(items) >= limit:
                break
            row = ds[idx]
            gold = label_names[row["label"]]
            fmt = format_decision_row(
                tokenizer,
                state=row["text"],
                question="Which intent does this banking inquiry express?",
                criteria=criteria,
                gold_option=gold,
                max_len=1024,  # Menampung 77 opsi
            )
            if fmt and fmt["label"] != -1:
                items.append(fmt)

    elif bench_name == "clinc150":
        ds = load_dataset("clinc/clinc_oos", "plus", split="test")
        intent_names = ds.features["intent"].names
        criteria = {intent: intent.replace("_", " ") for intent in intent_names}
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            if len(items) >= limit:
                break
            row = ds[idx]
            gold = intent_names[row["intent"]]
            fmt = format_decision_row(
                tokenizer,
                state=row["text"],
                question="Classify the intent of this customer message:",
                criteria=criteria,
                gold_option=gold,
                max_len=1024,  # Menampung opsi besar
            )
            if fmt and fmt["label"] != -1:
                items.append(fmt)

    print(f"Benchmark '{bench_name}': berhasil menyiapkan {len(items)} baris uji.", flush=True)
    return items


# ==============================================================================
# 4. Kalibrasi Pasca-Latih & Metrik (ECE / Brier / Akurasi)
# ==============================================================================
def calibrate_model_temperature(
    model: OptionMarkerDecisionModel,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Optimasi temperatur skalar T via L-BFGS pada NLL data validasi."""
    model.eval()
    all_row_logits, all_row_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pos = batch["mask_positions"].to(device)
            val = batch["mask_valid"].to(device)
            out = model(inp, mask, pos, val)
            logits = out["logits"].cpu()
            valid = batch["mask_valid"].cpu()
            labels = batch["labels"].cpu()

            for b_i in range(logits.size(0)):
                n_opts = int(valid[b_i].sum().item())
                all_row_logits.append(logits[b_i, :n_opts])
                all_row_labels.append(int(labels[b_i].item()))

    temp_param = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temp_param], lr=0.02, max_iter=50)

    def closure():
        optimizer.zero_grad()
        T = temp_param.clamp(min=0.05, max=10.0)
        loss = sum(
            F.cross_entropy((z / T).unsqueeze(0), torch.tensor([g]))
            for z, g in zip(all_row_logits, all_row_labels)
        ) / max(len(all_row_logits), 1)
        loss.backward()
        return loss

    optimizer.step(closure)
    best_temp = float(temp_param.detach().clamp(min=0.05, max=10.0).item())
    with torch.no_grad():
        model.log_temp.copy_(torch.tensor(math.log(best_temp), device=model.log_temp.device))
    return best_temp


def evaluate_dataset(
    model: OptionMarkerDecisionModel,
    data_loader: DataLoader,
    device: torch.device,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Menghitung Akurasi dan Expected Calibration Error (ECE)."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pos = batch["mask_positions"].to(device)
            val = batch["mask_valid"].to(device)
            labels = batch["labels"]

            out = model(inp, mask, pos, val)
            probs = out["probabilities"].cpu()
            max_p, preds = probs.max(dim=-1)

            all_probs.extend(max_p.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    corrects = [int(p == l) for p, l in zip(all_preds, all_labels)]
    acc = sum(corrects) / max(len(corrects), 1)

    # Kalkulasi ECE (Expected Calibration Error)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total_samples = len(all_probs)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = [j for j, p in enumerate(all_probs) if bin_lower <= p < bin_upper]
        if in_bin:
            bin_acc = sum(corrects[j] for j in in_bin) / len(in_bin)
            bin_conf = sum(all_probs[j] for j in in_bin) / len(in_bin)
            ece += (len(in_bin) / total_samples) * abs(bin_acc - bin_conf)

    return {"accuracy": acc, "ece": ece, "total": total_samples}


# ==============================================================================
# 5. Pipeline Utama: Train, Calibrate, Eval, Save, Push ke HF Hub
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Colab Decision Model Fine-Tuning & Eval")
    parser.add_argument("--model_id", type=str, default="answerdotai/ModernBERT-base",
                        help="HuggingFace model ID (misal: answerdotai/ModernBERT-base, sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--eval_limit", type=int, default=500, help="Jumlah sampel uji per benchmark")
    parser.add_argument("--output_dir", type=str, default="out-decision-encoder")
    parser.add_argument("--hf_repo", type=str, default=os.environ.get("HF_REPO", ""))
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan Device: {device} | Model: {args.model_id}", flush=True)

    # 1. Inisialisasi Tokenizer & Backbone
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if not tokenizer.mask_token:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    backbone = AutoModel.from_pretrained(args.model_id)
    hidden_size = backbone.config.hidden_size
    model = OptionMarkerDecisionModel(backbone, hidden_size=hidden_size).to(device)

    # 2. Persiapan Data Train & Val
    train_data, val_data = load_and_prep_datasets(tokenizer, seed=args.seed, max_train_samples=args.max_train_samples)
    pad_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        ListDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        ListDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # 3. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    print(f"\n--- Memulai Training ({args.epochs} Epoch, {total_steps} Steps) ---", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pos = batch["mask_positions"].to(device)
            val = batch["mask_valid"].to(device)
            labels = batch["labels"].to(device)

            out = model(inp, mask, pos, val, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = out["probabilities"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}", flush=True)

        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {total_loss / len(train_loader):.4f} | Acc: {correct / total * 100:.2f}% | Durasi: {dt:.1f}s", flush=True)

    # 4. Kalibrasi Suhu Pasca-Latih
    print("\n--- Melakukan Post-Hoc Temperature Calibration (L-BFGS) ---", flush=True)
    calibrated_t = calibrate_model_temperature(model, val_loader, device)
    print(f"Suhu terkalibrasi: T = {calibrated_t:.3f}")

    # 5. Evaluasi 3 Benchmark Standar
    print("\n--- Menjalankan Evaluasi 3 Benchmark (Typed Decisions, BANKING77, CLINC150) ---", flush=True)
    benches = ["typed_decisions", "banking77", "clinc150"]
    bench_results = {}

    for bname in benches:
        test_items = load_benchmark_test_set(bname, tokenizer, limit=args.eval_limit, seed=args.seed)
        if not test_items:
            bench_results[bname] = {"accuracy": 0.0, "ece": 0.0}
            continue
        test_loader = DataLoader(
            ListDataset(test_items),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )
        metrics = evaluate_dataset(model, test_loader, device)
        bench_results[bname] = metrics
        print(f"[{bname.upper()}] Akurasi: {metrics['accuracy'] * 100:.2f}% | ECE: {metrics['ece']:.4f}")

    # Tampilkan Tabel Hasil
    print("\n" + "=" * 65)
    print("| Benchmark Dataset | Akurasi | ECE (Kalibrasi Error) | Sampel |")
    print("| :--- | :---: | :---: | :---: |")
    for bname, m in bench_results.items():
        print(f"| {bname:17} | {m['accuracy']*100:6.2f}% | {m['ece']:17.4f} | {m.get('total', 0):6} |")
    print("=" * 65 + "\n")

    # 6. Simpan Model & Artefak Lokal
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Menyimpan model ke direktori lokal: {args.output_dir}...")
    tokenizer.save_pretrained(args.output_dir)
    model.backbone.save_pretrained(args.output_dir)

    # Simpan bobot head & konfigurasi kalibrasi
    head_payload = {
        "head_state_dict": model.head.state_dict(),
        "temperature": calibrated_t,
        "hidden_size": hidden_size,
        "model_id": args.model_id,
        "bench_results": bench_results,
    }
    torch.save(head_payload, os.path.join(args.output_dir, "decision_head.pt"))

    with open(os.path.join(args.output_dir, "decision_meta.json"), "w") as f:
        json.dump(head_payload, f, indent=2, default=str)

    print("Model dan decision head berhasil disimpan.")

    # 7. Push ke Hugging Face Hub (Opsional jika token & repo diberikan)
    if args.hf_repo and args.hf_token:
        print(f"\n--- Mengunggah Model ke Hugging Face Hub: {args.hf_repo} ---", flush=True)
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.hf_token)
            api.create_repo(repo_id=args.hf_repo, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=args.output_dir,
                repo_id=args.hf_repo,
                commit_message=f"Upload System One Decision Model fine-tuned from {args.model_id}",
            )
            print(f"Sukses! Model kini tersedia publik di https://huggingface.co/{args.hf_repo}")
        except Exception as e:
            print(f"Peringatan: Gagal mengunggah ke Hugging Face ({e})")
    else:
        print("HF_REPO atau HF_TOKEN tidak disetel. Model hanya tersimpan lokal.")


if __name__ == "__main__":
    main()
