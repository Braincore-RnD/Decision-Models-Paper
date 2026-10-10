Untuk melakukan reverse engineering dan membangun model keputusan System One seperti Laya dari nol (from scratch), Anda harus bergeser dari paradigma Autoregressive LLM (yang memprediksi kata per kata) ke paradigma Bidirectional Encoder + Structured Classification Head.
Berikut adalah cetak biru teknis lengkap yang mencakup arsitektur, metode training (RLCD), dan struktur data untuk membangunnya secara mandiri.

1. Arsitektur Model Laya (Tanpa Generasi Teks)

Model Laya tidak menggunakan dekoder untuk mengetik teks jawaban. Sebagai gantinya, ia menggunakan Encoder dua arah yang di atasnya dipasang kepala keputusan khusus (decision head).

Mekanisme Tokenisasi & Input Sequence

Jika sebuah LLM biasa menerima input "Klasifikasikan email ini...", Laya menyusun seluruh elemen (State/Konteks, Pertanyaan, dan Opsi Pilihan) ke dalam satu baris sequence menggunakan token khusus [MASK] sebagai jangkar nilai:
text
[CLS] Jenis_Tugas + Instruksi Pertanyaan [SEP] [MASK] Opsi_A [MASK] Opsi_B [MASK] Opsi_C [SEP] Teks_Konteks/State [SEP]
Gunakan kode dengan hati-hati.
• Fungsi [MASK]: Token mask ini diletakkan tepat sebelum deskripsi setiap opsi pilihan. Karena arsitekturnya bidirectional (berbasis ModernBERT-large), representasi vektor pada posisi [MASK] ini akan menyerap informasi kontekstual dari teks pertanyaan dan teks state secara bersamaan.

Arsitektur Kepala Keputusan (Decision Head)

Dari total 421 juta parameter Laya, sekitar 26 juta parameter dialokasikan untuk decision head yang dibangun dari nol:

1. Dua Lapisan Transformer Tambahan: Mengekstrak embedding representasi dari posisi token [MASK] yang disuplai.
2. Option-Marker Scorer: Memetakan vektor hidden state dari setiap [MASK] menjadi nilai skalar (logit) tunggal untuk opsi tersebut.
3. Act/Escalate Module: Komponen klasifikasi biner terpisah untuk menentukan apakah model memiliki keyakinan cukup tinggi untuk mengeksekusi tindakan atau harus melakukan escalation (mengoper masalah ke manusia).
Hasil akhirnya dilewatkan ke fungsi aktivasi (seperti Softmax untuk tipe Choice atau Sigmoid untuk tipe Noul) untuk menghasilkan nilai probabilitas terkalibrasi antara 0.0 hingga 1.0.

4. Apa itu RLCD (Reinforcement Learning for Calibrated Decisions)?

Masalah utama saat melatih model klasifikasi biasa adalah kecenderungan model menjadi overconfident. Model akan mendorong probabilitas opsi yang benar ke angka mendekati 1.0 demi meminimalkan Cross-Entropy Loss.
RLCD memecahkan ini dengan mengubah target optimasi: Hadiah (reward) tidak diberikan berdasarkan apakah tebakan model benar, melainkan apakah tingkat kepercayaan (probabilitas) yang dilaporkan model sesuai dengan kenyataan empiris.
Prinsip Kalibrasi: Jika model membuat 100 prediksi berbeda dengan tingkat kepercayaan masing-masing 80%, maka secara statistik harus ada tepat 80 prediksi yang benar dan 20 prediksi yang salah. Jika 100 prediksi tersebut ternyata benar semua (100% akurat), model tersebut dianggap terkalibrasi dengan buruk karena terlalu merendahkan kemampuan aslinya (underconfident).

Formula Reward: Strictly Proper Scoring Rules

Untuk mengimplementasikan RLCD dari nol, fungsi loss atau reward function agen PPO/Reinforcement Learning Anda harus menggunakan aturan penilaian matematis yang sensitif terhadap kalibrasi (Proper Scoring Rules):
• Logarithmic Score: Mengukur penalti informasi berdasarkan logaritma probabilitas opsi yang benar.
• Brier Score (Spherical/Quadratic Score): Mengukur jarak kuadrat terkecil antara vektor probabilitas prediksi dengan vektor aktual (one-hot vector).
• Ranked Probability Score (RPS): Digunakan khusus untuk tipe keputusan ordinal (Score) untuk menghukum model jika ia meleset jauh dari skala rating yang seharusnya.

1. Alur Pelatihan (Training Method) dari Nol

Untuk merekayasa balik model ini, ikuti metodologi pelatihan dua tahap (two-stage recipe) seperti yang diuraikan pada riset reproduksi model keputusan sejenis:
[Tahap 1: SFT] Base Encoder (ModernBERT) -> Latih Decision Head dengan Soft Targets (Cross-Entropy)
                                      |
                                      v
[Tahap 2: RLCD] Jalankan RL (PPO/GRPO) -> Optimasi Output Probabilitas Menggunakan Proper Scoring Rules

1. Tahap 1: Supervised Fine-Tuning (SFT) dengan Soft Targets
 • Hubungkan Base Encoder pilihan Anda (misal ModernBERT atau DeBERTa) dengan Decision Head acak yang Anda buat.
 • Latih model menggunakan dataset berlabel manusia. Jangan gunakan target hard label (0 atau 1) yang kaku. Gunakan distribusi soft targets (misal: agregasi dari penilaian beberapa annotator) untuk melatih sensitivitas awal model terhadap ambiguitas data.
2. Tahap 2: Fase RLCD (Alignment & Kalibrasi)
 • Masukkan state dan skema pertanyaan ke model, biarkan model melakukan sampling distribusi internalnya.
 • Hitung nilai Proper Scoring Rules (misal Brier Score) dari keputusan yang diambil.
 • Masukkan skor tersebut sebagai nilai Reward ke dalam algoritma RL (seperti PPO atau GRPO). Fase ini akan menekan ketidakpastian model dan memaksa distribusi logitnya mencerminkan probabilitas realitas.
3. Tahap Akhir: Temperature Calibration
 • Lakukan kalibrasi temperatur pasca-pelatihan (post-processing calibration) yang disesuaikan dengan jumlah opsi pilihan (per-option-count temperature calibration) untuk memastikan nilai skalar output tetap stabil baik saat dihadapkan pada 2 pilihan maupun 10 pilihan sekaligus.

Kode Purwarupa: Menghitung Skor Kalibrasi (Brier Score)

Saat menulis kode loss function kustom Anda di PyTorch, modul evaluasi kalibrasi dasarnya akan terlihat seperti ini:
python
import torch
import torch.nn.functional as F

def brier_score_loss(prob_predictions, target_labels, num_classes):
    """
    Menghitung Brier Score sebagai proper scoring rule untuk RLCD.
    prob_predictions: Tensor probabilitas hasil softmax model [Batch, Num_Classes]
    target_labels: Tensor indeks jawaban yang benar [Batch]
    """
    # Ubah target menjadi one-hot vector
    target_one_hot = F.one_hot(target_labels, num_classes=num_classes).float()

    # Hitung mean squared error antara probabilitas prediksi dan kenyataan
    # Nilai semakin mendekati 0 berarti kalibrasi semakin sempurna
    brier_score = torch.mean(torch.sum((prob_predictions - target_one_hot) ** 2, dim=1))
    return brier_score
Gunakan kode dengan hati-hati.
Untuk mulai membangun repositori scratch Anda, bagian mana yang ingin Anda rancang terlebih dahulu:
• Struktur data loader / penyiapan token masker di PyTorch?
• Kode integrasi loss function RLCD untuk agen RL?
• Desain arsitektur dua lapis kepala transformer untuk klasifikasinya?
Mari kita bedah kode spesifiknya bersama-sama.
