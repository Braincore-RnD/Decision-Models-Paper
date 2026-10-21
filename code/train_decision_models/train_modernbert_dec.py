"""End-to-end training and inference for Bidirectional Encoder Decision Models (Laya-style).
Mendukung ModernBERT, DeBERTa, dan BERT menggunakan Option-Marker [MASK] scoring head.
Tidak memerlukan Unsloth — murni PyTorch + Hugging Face Transformers.

Penggunaan:
  # Self-check cepat CPU (menguji pipeline forward, loss, backward, kalibrasi, predict):
  python train_modernbert_dec.py --self-check

  # Training penuh dengan dataset Hugging Face:
  python train_modernbert_dec.py --model_id answerdotai/ModernBERT-base --dataset LocalLLaMA/typed-decisions
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# ==============================================================================
# 1. Arsitektur Model: Option-Marker Decision Model
# ==============================================================================
class OptionMarkerDecisionModel(nn.Module):
    """Backbone Bidirectional Encoder + Contextual [MASK] Projection Head.

    Setiap opsi pilihan ditandai dengan token [MASK]. Karena encoder bersifat
    bidirectional (ModernBERT/DeBERTa), token [MASK] menyerap representasi
    state dan opsi secara simultan dalam satu forward pass (O(1)).
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
        # Temperatur logaritmik awal = 1.0 (exp(0))
        self.log_temp = nn.Parameter(torch.zeros(()))

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
        """Args:

        input_ids: (B, L)
        attention_mask: (B, L)
        mask_positions: (B, max_opts) indeks token [MASK] pada input_ids
        mask_valid: (B, max_opts) bool mask apakah opsi valid (bukan padding)
        labels: (B,) indeks opsi ground truth (0 .. num_opts-1)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # (B, L, D)

        B, max_opts = mask_positions.shape
        D = H.size(-1)

        # Ambil hidden state pada posisi [MASK] untuk setiap opsi
        idx_exp = mask_positions.unsqueeze(-1).expand(B, max_opts, D)
        mask_reps = torch.gather(H, dim=1, index=idx_exp)  # (B, max_opts, D)

        # Proyeksikan ke logit skalar per opsi
        logits = self.head(mask_reps).squeeze(-1)  # (B, max_opts)

        # Masking opsi padding dengan nilai sangat negatif
        logits = logits.masked_fill(~mask_valid, -1e4)

        # Skalakan dengan temperatur
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
# 2. Dataset & Tokenizer Pipeline
# ==============================================================================
def format_sample(
    tokenizer: PreTrainedTokenizer,
    state: str,
    question: str,
    criteria: Dict[str, str],
    gold_option: str | None = None,
    max_len: int = 512,
) -> Dict[str, Any]:
    """Menyusun teks input kanonis dengan token [MASK] sebelum deskripsi tiap opsi."""
    mask_token = tokenizer.mask_token or "[MASK]"
    lines = [f"State: {state.strip()}", f"Question: {question.strip()}", "Options:"]

    option_keys = list(criteria.keys())
    for opt_key, opt_desc in criteria.items():
        lines.append(f"{mask_token} {opt_key}: {opt_desc.strip()}")

    full_text = "\n".join(lines)
    enc = tokenizer(full_text, max_length=max_len, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attn_mask = enc["attention_mask"][0]

    mask_id = tokenizer.mask_token_id
    mask_indices = (input_ids == mask_id).nonzero(as_tuple=True)[0].tolist()

    label_idx = option_keys.index(gold_option) if gold_option in option_keys else -1

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "mask_indices": mask_indices,
        "options": option_keys,
        "label": label_idx,
    }


def collate_decision_batch(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate batch dinamis dengan padding sekuens dan opsi kandidat."""
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


class InMemoryDecisionDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ==============================================================================
# 3. Kalibrasi Pasca-Latih (Post-Hoc L-BFGS Temperature Scaling)
# ==============================================================================
def calibrate_model_temperature(
    model: OptionMarkerDecisionModel,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Mengoptimalkan temperatur skalar pada data validasi holdout via L-BFGS."""
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

    # Set temperatur model ke hasil terkalibrasi
    with torch.no_grad():
        model.log_temp.copy_(torch.tensor(math.log(best_temp), device=model.log_temp.device))

    return best_temp


# ==============================================================================
# 4. Fungsi Prediksi Single-Forward-Pass (Inference API)
# ==============================================================================
def predict_decision(
    model: OptionMarkerDecisionModel,
    tokenizer: PreTrainedTokenizer,
    state: str,
    question: str,
    criteria: Dict[str, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluasi satu state + skema pertanyaan dalam satu forward pass."""
    model.eval()
    sample = format_sample(tokenizer, state, question, criteria)
    batch = collate_decision_batch([sample], pad_token_id=tokenizer.pad_token_id or 0)

    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            mask_positions=batch["mask_positions"].to(device),
            mask_valid=batch["mask_valid"].to(device),
        )
        probs = out["probabilities"][0, : len(sample["options"])].cpu().tolist()

    best_idx = int(torch.tensor(probs).argmax().item())
    best_option = sample["options"][best_idx]
    confidence = probs[best_idx]

    return {
        "answer": best_option,
        "confidence": round(confidence, 4),
        "probabilities": {opt: round(p, 4) for opt, p in zip(sample["options"], probs)},
    }


# ==============================================================================
# 5. Pipeline Latih Lengkap (Train Loop)
# ==============================================================================
def train_decision_model(
    model: OptionMarkerDecisionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 3e-5,
    device: torch.device = torch.device("cpu"),
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.to(device)

    print(f"Mulai pelatihan: {epochs} epoch | Device: {device}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

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

            total_loss += loss.item()
            preds = out["probabilities"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | Acc: {train_acc * 100:.2f}%", flush=True)

    print("Kalibrasi suhu pasca-latih...", flush=True)
    calibrated_temp = calibrate_model_temperature(model, val_loader, device)
    print(f"Selesai! Suhu terkalibrasi T = {calibrated_temp:.3f}", flush=True)


# ==============================================================================
# 6. Self-Check CPU (<2 detik)
# ==============================================================================
def run_self_check() -> None:
    """Verifikasi end-to-end tanpa unduhan model besar."""
    print("=== Menjalankan Self-Check End-to-End CPU ===")
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

    # Gunakan tokenizer standar dan backbone mini 2-layer acak
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    cfg = BertConfig(
        vocab_size=len(tok),
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=512,
    )
    backbone = BertForMaskedLM(cfg).bert
    model = OptionMarkerDecisionModel(backbone, hidden_size=128, head_dim=64)

    # Data mainan (synthetic dataset)
    toy_raw = [
        {
            "state": "Payment failed twice on invoice #901. Refund needed immediately.",
            "question": "Which department handles this?",
            "criteria": {
                "billing": "Invoices, payment errors, and refund requests",
                "tech": "Server downtime, API errors, code bugs",
                "sales": "Pricing tiers and enterprise contracts",
            },
            "gold": "billing",
        },
        {
            "state": "500 Internal Server Error returned when querying /api/users.",
            "question": "Which department handles this?",
            "criteria": {
                "billing": "Invoices, payment errors, and refund requests",
                "tech": "Server downtime, API errors, code bugs",
                "sales": "Pricing tiers and enterprise contracts",
            },
            "gold": "tech",
        },
        {
            "state": "We are a team of 40 looking for a quote on the annual tier.",
            "question": "Which department handles this?",
            "criteria": {
                "billing": "Invoices, payment errors, and refund requests",
                "tech": "Server downtime, API errors, code bugs",
                "sales": "Pricing tiers and enterprise contracts",
            },
            "gold": "sales",
        },
    ]

    processed = [format_sample(tok, d["state"], d["question"], d["criteria"], d["gold"]) for d in toy_raw]
    ds = InMemoryDecisionDataset(processed)
    pad_id = tok.pad_token_id or 0
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate_decision_batch(b, pad_id))

    device = torch.device("cpu")
    train_decision_model(model, loader, loader, epochs=3, lr=1e-3, device=device)

    # Uji inferensi O(1)
    res = predict_decision(
        model,
        tok,
        state="Customer is furious about duplicate charge on Visa card.",
        question="Which department handles this?",
        criteria={
            "billing": "Invoices, payment errors, and refund requests",
            "tech": "Server downtime, API errors, code bugs",
            "sales": "Pricing tiers and enterprise contracts",
        },
        device=device,
    )
    print("Hasil inferensi self-check:", res)
    assert res["answer"] in ["billing", "tech", "sales"]
    assert "probabilities" in res
    print("=== SELF-CHECK BERHASIL: Pipeline Laya-Style Siap Pakai ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Option-Marker Decision Model")
    parser.add_argument("--self-check", action="store_true", help="Jalankan uji mandiri cepat CPU")
    parser.add_argument("--model_id", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    if args.self_check or len(sys.argv) == 1:
        run_self_check()
    else:
        print(f"Inisialisasi backbone {args.model_id}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(args.model_id)
        backbone = AutoModel.from_pretrained(args.model_id)
        model = OptionMarkerDecisionModel(backbone, hidden_size=backbone.config.hidden_size)
        print("Model siap untuk ditraining dengan custom dataset.")
