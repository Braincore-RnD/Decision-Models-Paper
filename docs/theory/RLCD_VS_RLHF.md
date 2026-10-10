---
title: "RLCD vs RLHF"
date: 2026-09-20
tags:
  - rlcd
  - rlhf
  - comparison
  - calibration
  - decision-models
aliases:
  - RLCD VS RLHF
---

> Terkait: [[00-index-rlcd-and-system-one|Index RLCD]] · [[02-rlcd-vs-rlhf-comparison|Perbandingan Mendalam]] · [[01-rlcd-theory-and-formalism|Teori RLCD]] · [[07-evaluation-methodology-and-benchmarks|Evaluasi & Benchmark]] · [[03-typesafe-jev-architecture|Arsitektur TypeSafe Jev]] · [[04-laya-decision-engine|Laya Decision Engine]]

Saya sudah cross-check paper, model card, repository, dan materi resmi yang tersedia publik. Ada satu hal yang sangat penting untuk dijadikan starting point:

**RLCD saat ini belum merupakan nama algoritme RL yang sudah memiliki formulasi akademik baku seperti PPO, DPO, atau GRPO.** TypeSafe memperkenalkan “Reinforcement Learning for Calibrated Decisions” sebagai nama pendekatan yang digunakan Jev, tetapi detail optimizer, reward lengkap, sampling procedure, dataset, dan ablation Jev belum dipublikasikan secara teknis. Sebaliknya, Laya memberikan implementasi open-source yang jauh lebih konkret, sehingga Laya justru menjadi contoh paling berguna untuk merekonstruksi teori RLCD secara matematis. ([TypeSafe AI][1])

Lebih menarik lagi, GLiNER2.5 berada di titik yang sangat dekat secara arsitektur tetapi berbeda secara objective. Ia adalah bidirectional encoder, schema-driven, non-autoregressive, dan menghasilkan structured decisions, tetapi paper-nya mendeskripsikan supervised/multi-task training, bukan RLCD. ([arXiv][2])

Jadi saya akan membangun riset ini dari tiga layer:

**architecture → objective → optimization**

Ini penting karena "non-autoregressive", "structured output", dan "RLCD" bukan tiga nama untuk hal yang sama.

---

# 1. Apa sebenarnya RLCD?

Secara konseptual:

RLHF bertanya:

> “Output mana yang lebih disukai manusia?”

RLVR bertanya:

> “Output mana yang dapat diverifikasi benar?”

RLCD bertanya:

> “Apakah distribusi probabilitas yang diberikan model sesuai dengan outcome yang benar-benar terjadi?”

TypeSafe sendiri menempatkan ketiganya seperti itu dalam pengenalan Jev. ([TypeSafe AI][1])

Perbedaan fundamentalnya adalah **unit optimisasinya**.

Pada RLHF, model biasanya menghasilkan string:

$$
y=(y_1,y_2,\ldots,y_T)
$$

kemudian reward diberikan kepada keseluruhan completion.

Pada RLCD, output bisa berupa distribution:

$$
q_\theta(y|x)
=
[q_1,q_2,\ldots,q_K]
$$

dengan

$$
\sum_{k=1}^{K}q_k=1
$$

dan target sebenarnya:

$$
y^* \in \{1,\ldots,K\}
$$

Jadi model bukan sekadar mengatakan:

> “Saya memilih billing.”

tetapi:

$$
P(\text{billing})=0.82
$$

$$
P(\text{support})=0.12
$$

$$
P(\text{technical})=0.06
$$

Dan ketika data outcome datang, sistem memberikan reward berdasarkan **kualitas distribusi tersebut**, bukan hanya apakah argmax-nya benar.

Itulah inti matematis RLCD.

---

# 2. Calibration sebenarnya apa?

Misalkan model selalu menghasilkan:

$$
q=P(Y=1|X)
$$

Sebuah classifier disebut calibrated apabila:

$$
P(Y=1 \mid q(X)=p)=p
$$

Artinya kalau model mengatakan:

> probability = 0.8

maka, pada populasi prediksi yang mendapatkan angka sekitar 0.8, kira-kira 80% memang benar.

Bukan berarti setiap prediksi 0.8 harus benar 80%. Calibration adalah **property populasi**, bukan guarantee per-instance. ([Springer][3])

Untuk multiclass:

$$
P(Y=k \mid Q=q)=q_k
$$

Jadi:

$$
q=[0.7,0.2,0.1]
$$

berarti kelompok prediction dengan distribusi seperti itu harus secara empiris memiliki outcome kira-kira 70%, 20%, 10%.

Ini berbeda dari accuracy.

Contoh:

Model A:

$$
q=[0.51,0.49]
$$

Model B:

$$
q=[0.99,0.01]
$$

dan jawaban ternyata class 1.

Keduanya punya accuracy:

$$
Accuracy=1
$$

tetapi secara epistemic behavior keduanya sangat berbeda.

Model A berkata:

> "Saya hampir tidak yakin."

Model B berkata:

> "Saya hampir pasti."

Jika model B ternyata salah pada 10% kasus serupa, B punya probabilistic behavior yang baik. Kalau salah 40%, B sangat overconfident.

Jadi RLCD mencoba mengoptimalkan **bukan hanya classification correctness, tetapi probabilistic honesty**.

---

# 3. Kenapa proper scoring rule menjadi fondasi RLCD?

Ini bagian yang paling menarik secara teori.

Proper scoring rule adalah fungsi:

$$
S(q,y)
$$

yang memberi reward kepada forecaster berdasarkan probability distribution \(q\) dan outcome aktual \(y\).

Ia disebut proper apabila:

$$
\mathbb{E}_{y\sim p}[S(p,y)]
\geq
\mathbb{E}_{y\sim p}[S(q,y)]
$$

untuk semua \(q\).

Dan **strictly proper** apabila equality hanya terjadi ketika:

$$
q=p
$$

Jadi satu-satunya strategi optimal adalah melaporkan probabilitas yang benar. Ini adalah fondasi klasik dalam probabilistic forecasting. ([DOI System][4])

---

# 4. Logarithmic score

Salah satu strictly proper scoring rule paling penting:

$$
S_{\log}(q,y)
=
\log q_y
$$

Misalkan true distribution:

$$
p=[p_1,\ldots,p_K]
$$

Expected reward:

$$
\mathbb{E}_{y\sim p}
[\log q_y]
=
\sum_k p_k\log q_k
$$

Tambahkan dan kurangkan \(p_k\log p_k\):

$$
=
\sum_kp_k\log p_k
-
\sum_kp_k
\log\frac{p_k}{q_k}
$$

sehingga:

$$
\mathbb{E}[S_{\log}]
=
-H(p)-D_{KL}(p\|q)
$$

karena:

$$
D_{KL}(p\|q)\geq0
$$

maka expected reward maksimum ketika:

$$
D_{KL}(p\|q)=0
$$

yang berarti:

$$
\boxed{q=p}
$$

Inilah alasan log score sangat powerful.

Menariknya, log-likelihood/cross-entropy yang kita gunakan dalam supervised classification sebenarnya sudah merupakan **strictly proper scoring rule**. ([Proceedings of Machine Learning Research][5])

Ini melahirkan satu pertanyaan penelitian yang sangat penting:

> **Kalau cross-entropy sudah mathematically incentivize calibrated probabilities, kenapa kita perlu RLCD?**

Nanti ini akan menjadi salah satu poin terpenting.

---

# 5. Spherical score

Laya juga menyebut spherical score.

Bentuk sederhananya:

$$
S_{sph}(q,y)
=
\frac{q_y}{\sqrt{\sum_k q_k^2}}
$$

Dengan vector form:

$$
S_{sph}(q,y)
=
\frac{e_y^Tq}{\|q\|_2}
$$

Expected score:

$$
\mathbb{E}[S]
=
\frac{p^Tq}{\|q\|_2}
$$

Maximum terjadi ketika \(q\) memiliki arah yang sama dengan \(p\). Karena keduanya harus berada pada probability simplex, akhirnya:

$$
q=p
$$

Jadi spherical score juga strictly proper. Literature modern tentang proper scoring rules membahas log, spherical, quadratic/Brier, dan berbagai family lainnya. ([DOI System][4])

---

# 6. Brier score

Untuk binary prediction:

$$
S_{Brier}(q,y)
=
-(q-y)^2
$$

atau sebagai loss:

$$
L_{Brier}=(q-y)^2
$$

Untuk multiclass:

$$
L_{Brier}
=
\sum_k(q_k-y_k)^2
$$

dengan \(y\) sebagai one-hot vector.

Brier score juga proper.

Ini menarik karena Brier lebih bounded dibanding log score.

Artinya jika:

$$
q\rightarrow0
$$

sementara:

$$
y=1
$$

log loss menjadi:

$$
-\log q\rightarrow\infty
$$

sedangkan Brier hanya sebesar:

$$
(0-1)^2=1
$$

Karena itu Brier sering menarik untuk reinforcement learning yang sensitif terhadap reward scale.

---

# 7. Ranked Probability Score

Untuk ordinal task seperti:

$$
0=\text{low}
$$

$$
1=\text{medium}
$$

$$
2=\text{high}
$$

$$
3=\text{critical}
$$

accuracy biasa kurang ideal karena kesalahan:

$$
0\rightarrow1
$$

tidak sama dengan:

$$
0\rightarrow3
$$

RPS menggunakan cumulative distribution:

$$
F_q(k)=\sum_{j=1}^{k}q_j
$$

kemudian:

$$
RPS(q,y)
=
\sum_{k=1}^{K-1}
(F_q(k)-F_y(k))^2
$$

Sebagai reward kita memakai:

$$
S_{RPS}=-RPS
$$

Laya memang menyebut penggunaan RPS untuk ordinal score questions. ([Hugging Face][6])

---

# 8. Composite RLCD reward

Implementasi Laya secara eksplisit menyebut:

$$
Reward =
\text{log score}
+
\text{spherical score}
+
\text{ranked probability score}
$$

untuk task sesuai tipenya. ([Hugging Face][6])

Secara umum kita bisa menulis:

$$
R(q,y)
=
\alpha S_{\log}(q,y)
+
\beta S_{sph}(q,y)
-
\gamma RPS(q,y)
$$

dengan:

$$
\alpha,\beta,\gamma >0
$$

Karena positive combination dari strictly proper scoring rules tetap proper, objective ini tetap memiliki optimum pada true conditional distribution, dengan asumsi data/outcome memang mewakili target probabilitas yang ingin dipelajari.

Ini adalah alasan teoritis mengapa RLCD bukan sekadar:

> "Tambahkan confidence head."

Confidence benar-benar dimasukkan ke objective.

---

# 9. Di mana RL-nya masuk?

Ini bagian yang sangat menarik.

Misalkan model memiliki logits:

$$
z_\theta(x)
\in\mathbb{R}^{K}
$$

Kemudian:

$$
q_\theta
=
softmax(z_\theta)
$$

Laya menjelaskan bahwa exploration dilakukan dengan menambahkan Gaussian noise ke logits. ([Hugging Face][6])

Jadi:

$$
\tilde z
=
z_\theta(x)+\epsilon
$$

dengan:

$$
\epsilon\sim\mathcal N(0,\sigma^2I)
$$

kemudian:

$$
q_\theta^\epsilon
=
softmax(\tilde z)
$$

Setiap stochastic rollout memberikan:

$$
q_1,q_2,\ldots,q_G
$$

kemudian setiap prediction mendapatkan reward:

$$
R_i=S(q_i,y)
$$

Ini berbeda dari supervised learning biasa karena model sekarang memiliki exploration distribution.

---

# 10. REINFORCE + group baseline

Laya multilingual menyebut secara eksplisit:

> pure policy-gradient REINFORCE with GRPO-style group baseline. ([Hugging Face][7])

Secara konseptual, kita bisa menulis:

$$
A_i
=
\frac{R_i-\mu_R}
{\sigma_R+\epsilon}
$$

dengan:

$$
\mu_R=\frac1G\sum_{i=1}^{G}R_i
$$

dan loss:

$$
L_{RLCD}
=
-\frac1G
\sum_{i=1}^{G}
A_i\log\pi_\theta(a_i|x)
$$

Ini sangat mirip dengan philosophy GRPO:

> tidak perlu learned critic secara eksplisit, gunakan relative quality terhadap sample group.

Namun ada catatan penting secara akademik.

**Formula lengkap RLCD milik TypeSafe tidak dipublikasikan.**

Formula di atas adalah rekonstruksi matematis yang konsisten dengan implementasi Laya yang dipublikasikan, bukan klaim bahwa ini adalah exact internal formula Jev.

---

# 11. Ini membuat RLCD berbeda dari PPO

Pada RLHF klasik:

$$
x\rightarrow y
$$

Kemudian:

$$
r_\phi(x,y)
$$

dari reward model.

Lalu PPO melakukan:

$$
L^{PPO}
=
\mathbb E
[
\min(
r_t(\theta)A_t,
clip(r_t(\theta),1-\epsilon,1+\epsilon)A_t
)
]
$$

dengan:

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
$$

RLCD tidak harus memakai PPO.

Itu justru salah satu misconception penting.

**RLCD adalah objective/reward philosophy.**

Sementara:

* PPO = optimization algorithm
* REINFORCE = optimization algorithm
* GRPO = optimization algorithm
* proper scoring rule = reward/objective
* calibration = desired statistical property

Jadi secara taxonomy:

$$
\boxed{
RLCD \neq PPO
}
$$

dan:

$$
\boxed{
RLCD \neq GRPO
}
$$

Laya saat ini menggunakan RLCD + REINFORCE/group baseline. ([Hugging Face][7])

---

# 12. RLHF secara matematis

Dalam RLHF klasik, pipeline InstructGPT kira-kira:

$$
Pretrained\ LM
\rightarrow SFT
\rightarrow Reward\ Model
\rightarrow PPO
$$

OpenAI menjelaskan bahwa mereka mengumpulkan human-written demonstrations, kemudian human preference comparisons, melatih reward model, lalu mengoptimalkan policy dengan PPO. ([OpenAI][8])

Reward model:

$$
r_\phi(x,y)
$$

biasanya dilatih dari preference:

$$
y_w \succ y_l
$$

dengan:

$$
P(y_w\succ y_l|x)
=
\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

dan loss:

$$
L_{RM}
=
-\log
\sigma(
r_\phi(x,y_w)-r_\phi(x,y_l)
)
$$

Kemudian policy optimization:

$$
J(\theta)
=
\mathbb E[r_\phi(x,y)]
-
\beta
D_{KL}
(
\pi_\theta
\|
\pi_{ref}
)
$$

Jadi reward-nya adalah:

> human preference approximation.

RLCD mengganti reward tersebut menjadi:

> outcome-based probabilistic scoring.

---

# 13. Data RLHF vs RLCD

Perbedaannya cukup fundamental.

| Aspect             | RLHF                  | RLCD                                |
| ------------------ | --------------------- | ----------------------------------- |
| Input              | prompt                | state/question                      |
| Output             | text                  | probability distribution / decision |
| Label              | human preference      | observed outcome                    |
| Reward model       | biasanya diperlukan   | tidak wajib                         |
| Human ranking      | penting               | tidak harus                         |
| Reward             | preference            | proper scoring rule                 |
| Target             | preferred behavior    | calibrated probability              |
| Accuracy           | indirect              | direct                              |
| Calibration        | bukan objective utama | objective utama                     |
| Structured output  | bukan fokus           | native                              |
| Non-autoregressive | tidak diperlukan      | sangat cocok                        |

Jadi kebutuhan dataset juga berubah.

RLHF memerlukan:

$$
(x,y_w,y_l)
$$

sementara RLCD cukup memerlukan:

$$
(x,y)
$$

dengan:

$$
y=\text{observed outcome}
$$

Itu membuka use case yang sangat menarik.

Misalnya fraud detection:

$$
x=\text{transaction}
$$

dan outcome beberapa minggu kemudian:

$$
y\in\{fraud,legit\}
$$

Kita tidak memerlukan manusia untuk mengatakan transaction A lebih bagus daripada transaction B.

Kita hanya membutuhkan ground truth.

---

# 14. Tetapi bukankah cross-entropy sudah cukup?

**Ya. Dan ini mungkin research question paling penting dari seluruh topik ini.**

Misalkan task hanya:

$$
x\rightarrow y
$$

dengan multiclass label.

Cross entropy:

$$
L_{CE}
=
-\log q_\theta(y|x)
$$

sudah strictly proper.

Dalam expectation:

$$
\mathbb E[L_{CE}]
=
H(p)+D_{KL}(p\|q)
$$

minimum ketika:

$$
q=p
$$

Jadi secara asymptotic, ordinary supervised learning secara teori sudah menginginkan calibrated conditional distribution.

Karena itu, claim:

> “RLCD menjadi calibrated sedangkan supervised learning tidak”

tidak boleh diterima begitu saja.

Itu terlalu simplistik.

---

# 15. Lalu apa alasan RLCD tetap menarik?

Karena masalah dunia nyata sering bukan sekadar:

$$
x\rightarrow one\ label
$$

tetapi:

$$
state
\rightarrow
multiple\ decisions
\rightarrow
trajectory
\rightarrow
eventual\ outcome
$$

Misalnya:

Conversation:

$$
s_1\rightarrow s_2\rightarrow s_3\rightarrow s_4
$$

dan baru pada:

$$
s_4
$$

kita mengetahui:

$$
converted=1
$$

Model harus mengestimasi:

$$
P(Y=1|s_1)
$$

$$
P(Y=1|s_2)
$$

$$
P(Y=1|s_3)
$$

$$
P(Y=1|s_4)
$$

Di sini RL mulai mempunyai alasan lebih kuat.

---

# 16. Contoh SalesRLAgent

Ini adalah tahap awal riset ConvAI yang sangat menarik untuk dipelajari sebelum Laya.

Paper SalesRLAgent memformulasikan sales conversation sebagai sequential decision problem:

$$
state = conversation\ state
$$

$$
action = conversion\ probability
$$

$$
reward = prediction\ quality
$$

Mereka memakai lebih dari 1.2 juta synthetic conversations yang dibuat dengan GPT-4o, mencakup 15 industri, 3 sampai 27 turns, dengan median 8 turns. Data berisi transcript, speaker, outcome conversion, probability tiap turn, dan engagement metrics. ([arXiv][9])

Modelnya menggunakan Azure OpenAI embeddings 3072 dimensi sebagai bagian dari state representation, kemudian policy network, value network, dan uncertainty/meta-learning module. ([arXiv][9])

Training-nya mencakup supervised initialization lalu RL, curriculum, adversarial examples, dan ensemble techniques. ([arXiv][9])

Jadi evolusinya kira-kira:

$$
SalesRLAgent
\rightarrow
sequential\ probability\ prediction
$$

kemudian:

$$
Laya
\rightarrow
generalized\ structured\ probabilistic\ decision
$$

Ini lebih masuk akal sebagai historical progression daripada mengatakan Laya muncul dari nol.

Catatan penting, paper SalesRLAgent menggunakan synthetic GPT-4o generated data dan mengklaim 96.7% accuracy dalam evaluasinya, tetapi ini merupakan preprint dan methodological claims tersebut sebaiknya diperlakukan sebagai reported results, bukan sebagai established benchmark truth. ([arXiv][10])

---

# 17. Laya jauh lebih menarik sebagai reference implementation

Laya saat ini menyediakan implementasi yang paling terbuka di antara contoh yang Anda berikan.

Architecture:

$$
Input
\rightarrow
ModernBERT/mmBERT
\rightarrow
Decision\ Head
\rightarrow
Option\ logits
\rightarrow
Probability\ distribution
$$

Untuk English:

* ModernBERT-large, 395M backbone
* decision head
* total sekitar 421M
* 512 token budget

Sedangkan multilingual:

* mmBERT-base
* 22 layers
* hidden 768
* vocabulary 256k
* sekitar 322M total
* 1024 token context. ([Hugging Face][6])

Decision head menggunakan option markers:

$$
[MASK]_1,[MASK]_2,\ldots,[MASK]_K
$$

masing-masing mewakili candidate option.

Model kemudian menghasilkan:

$$
z_1,z_2,\ldots,z_K
$$

dan:

$$
p_k
=
\frac{e^{z_k/T}}
{\sum_j e^{z_j/T}}
$$

Ini sangat penting.

**Option space dapat berubah pada inference time.**

Berbeda dengan traditional fixed-label classifier:

$$
softmax(W h)
$$

di mana output head biasanya fixed terhadap training label vocabulary.

---

# 18. Data Laya

Untuk multilingual checkpoint, dokumentasinya menyebut dataset human-annotated real-world dari:

* multilingual intents
* NLI
* safety moderation
* prompt guardrails
* Enron spam
* phishing emails
* phishing URLs
* support tickets
* Bitext intents

dan checkpoint tersebut dilaporkan dilatih menggunakan RLCD dengan REINFORCE plus group baseline. ([Hugging Face][7])

Ini berbeda dengan SalesRLAgent yang memakai synthetic GPT-4o conversations.

---

# 19. Multi-turn RLCD dan TD(lambda)

Laya menyebut:

$$
TD(\lambda=1)
$$

untuk multi-turn dialogue prefix modeling. ([Hugging Face][6])

Secara teori:

$$
G_t^{(\lambda)}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}G_t^{(n)}
$$

Jika:

$$
\lambda=1
$$

maka kita mendekati Monte Carlo return:

$$
G_t
=
r_t+
\gamma r_{t+1}
+\cdots+
\gamma^{T-t}r_T
$$

Misalkan hanya final conversion yang diketahui:

$$
r_T=1
$$

dan sebelumnya:

$$
r_1=r_2=\cdots=r_{T-1}=0
$$

maka model dapat memberikan credit assignment ke prefix:

$$
s_1,s_2,\ldots,s_T
$$

tanpa memasukkan future conversation turns ke input.

Ini penting untuk menghindari leakage.

---

# 20. Laya typed-decisions lebih menarik lagi

Ada checkpoint khusus:

`convaiinnovations/laya-typed-decisions`

Yang dilatih dari 1,200-case training split atau sekitar 6,000 decisions.

Yang sangat menarik:

Laya tidak memakai RLCD secara murni.

Ia menggunakan:

$$
L
=
L_{RLCD}
+
\lambda L_{softCE}
$$

dengan soft cross-entropy terhadap teacher distributions. ([Hugging Face][11])

Artinya ini sebenarnya:

$$
\boxed{
RLCD + Knowledge\ Distillation
}
$$

Ini merupakan design pattern yang sangat layak diteliti.

RL memberikan:

> outcome-grounded probabilistic objective

Teacher distribution memberikan:

> richer local signal

Ini mirip filosofi:

$$
hard\ label + soft\ teacher + RL\ outcome
$$

daripada pure RL.

---

# 21. Temperature scaling

Ini juga sangat penting.

Misalkan raw logits:

$$
z
$$

Kita melakukan:

$$
q_k
=
softmax
\left(
\frac{z_k}{T}
\right)
$$

Interpretasi:

Jika:

$$
T>1
$$

distribution lebih flat.

Jika:

$$
T<1
$$

distribution lebih sharp.

Temperature optimal dapat dipelajari pada held-out validation data menggunakan NLL/Brier objective.

Dan dokumentasi Laya sendiri menunjukkan bahwa RLCD checkpoint belum otomatis perfectly calibrated. Misalnya mereka melaporkan post-hoc temperature fitting dapat menurunkan ECE secara signifikan pada beberapa checkpoint. ([Laya][12])

Ini membuktikan satu prinsip penting:

$$
\boxed{
RLCD\ training \neq guaranteed\ perfect\ calibration
}
$$

Finite data, model misspecification, distribution shift, optimization noise, dan label noise tetap ada.

---

# 22. GLiNER2.5: sangat relevan, tetapi bukan RLCD

Ini saya masukkan karena menurut saya justru sangat penting sebagai controlled comparison.

GLiNER2.5 menggunakan:

$$
Bidirectional\ Encoder
$$

dan schema-driven formulation:

$$
[Task\ Prompt]
\oplus
[SEP]
\oplus
[Input]
$$

Kemudian model melakukan entity extraction, classification, structured extraction, relations, dan task composition. ([arXiv][2])

GLiNER2.5 family sekarang mencakup:

* 74M small
* 194M base
* 287M multilingual

dengan boundary architecture. ([Hugging Face][13])

Training data pada paper GLiNER2 jauh berbeda:

$$
254,334\ examples
$$

yang terdiri atas:

$$
135,698
$$

real-world documents dan:

$$
118,636
$$

synthetic examples.

Real-world documents berasal dari news, Wikipedia, legal texts, PubMed, dan arXiv, lalu dianotasi GPT-4o. Synthetic examples juga dibuat menggunakan GPT-4o. ([arXiv][2])

Evaluation menggunakan:

SNIPS, Banking77, Amazon Intent, SST-2, IMDB, AG News, 20 Newsgroups, dan CrossNER. ([arXiv][2])

Jadi:

$$
GLiNER2.5
=
Schema-driven + Bidirectional Encoder
$$

sedangkan:

$$
Laya
=
Schema-driven + Bidirectional Encoder + RLCD
$$

Itulah controlled comparison yang sangat menarik untuk penelitian.

---

# 23. Bahkan sebenarnya GLiNER2.5 vs Laya bisa menjadi eksperimen ilmiah

Misalnya kita mengambil backbone yang sama:

$$
DeBERTa
$$

dan dataset yang sama.

Model A:

$$
L_{CE}
$$

Model B:

$$
L_{RLCD}
$$

Model C:

$$
L_{CE}+\lambda L_{RLCD}
$$

Model D:

$$
L_{CE}+\lambda L_{RLCD}+\mu L_{KD}
$$

Lalu bandingkan:

$$
Accuracy
$$

$$
NLL
$$

$$
Brier
$$

$$
ECE
$$

$$
AURC
$$

$$
Risk@Coverage
$$

$$
Selective\ Accuracy
$$

dan latency.

Nah, ini sudah menjadi **research program** yang jauh lebih kuat daripada sekadar mencoba "RLCD training".

---

# 24. Bagaimana dengan Qwen-2.5-1B-RLCD?

Ini justru harus diperlakukan hati-hati.

Model card `harshatheg/Qwen-2.5-1B-RLCD` memang memakai nama RLCD dan mendemonstrasikan parallel constrained decoding. Ia mengubah problem structured generation menjadi:

$$
context
\rightarrow
candidate\ tokens
\rightarrow
probabilities
$$

dengan temperature:

$$
P(c_i)
=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}
$$

dan melakukan candidate evaluation secara parallel menggunakan KV-cache broadcasting. ([Hugging Face][14])

Tetapi dokumentasi model tersebut tidak memberikan:

* RL training dataset
* reward formulation
* RL optimizer
* rollout count
* policy update
* RL training steps
* ablation

Sebaliknya, dokumentasinya fokus pada **parallel constrained decoding** dan menyebut model yang digunakan sebagai `Qwen2.5-1.5B-Instruct-4bit`. ([Hugging Face][14])

Jadi saya tidak akan memasukkan model ini sebagai bukti implementasi RLCD training.

Saya akan mengkategorikannya sebagai:

$$
\boxed{
RLCD\text{-}style\ probabilistic\ inference
}
$$

bukan:

$$
\boxed{
RLCD\text{-}trained\ model
}
$$

Ini perbedaan yang sangat penting.

---

# 25. LFM2.5-350M-RLCD bahkan lebih jelas

Model yang Anda berikan:

`notnotsamuel/LFM2.5-350M-RLCD`

secara eksplisit mengatakan:

> inference only, no training or fine-tuning

dan juga secara eksplisit mengatakan tidak mereproduksi training method proprietary Jev. ([Hugging Face][15])

Jadi ini bukan RLCD-trained model.

Ia adalah implementation eksperimen:

$$
LFM
\rightarrow
KV/cache reuse
\rightarrow
candidate scoring
\rightarrow
parallel structured inference
$$

yang memakai bounded candidate sets.

Sangat berguna untuk mempelajari **System 1 inference architecture**, tetapi bukan evidence tentang RLCD optimization.

---

# 26. Maka taxonomy model Anda sebenarnya seperti ini

| System               | Architecture             | Objective / Training                | RLCD evidence                     |
| -------------------- | ------------------------ | ----------------------------------- | --------------------------------- |
| InstructGPT          | Decoder LLM              | SFT + RM + PPO                      | No                                |
| Jev                  | Proprietary System One   | RLCD                                | Yes, vendor claim, details closed |
| SalesRLAgent         | Embedding + policy/value | PPO-style RL                        | precursor concept                 |
| Laya                 | Bidirectional encoder    | RLCD + REINFORCE                    | Yes, open                         |
| Laya Typed Decisions | Bidirectional encoder    | RLCD + soft CE                      | Yes, open                         |
| GLiNER2.5            | Bidirectional encoder    | Multi-task supervised               | No                                |
| Qwen-2.5-1B-RLCD     | Decoder LLM              | parallel constrained inference      | No documented RL training         |
| LFM2.5-350M-RLCD     | LFM                      | inference-only constrained decoding | explicitly no                     |

Itu menurut saya gambaran paling akurat dari evidence publik yang tersedia sekarang.

---

# 27. Kenapa RLCD cocok dengan System 1?

Ini sebenarnya bukan terutama tentang intelligence.

Ini tentang:

$$
latency
+
structured\ decisions
+
uncertainty
+
automation
$$

LLM biasa:

$$
text
\rightarrow
token_1
\rightarrow
token_2
\rightarrow
\cdots
\rightarrow
token_n
$$

System 1 decision model:

$$
text
\rightarrow
encoder
\rightarrow
distribution
$$

Jadi satu forward pass.

Contoh:

$$
P(\text{billing})=.87
$$

$$
P(\text{technical})=.09
$$

$$
P(\text{sales})=.04
$$

Business logic kemudian:

$$
P>0.85
\Rightarrow auto-route
$$

atau:

$$
P<0.85
\Rightarrow human\ review
$$

Inilah reason calibration menjadi sangat penting.

---

# 28. Dari probabilitas ke decision theory

Misalkan terdapat dua tindakan:

$$
a_1=\text{automate}
$$

$$
a_2=\text{human review}
$$

dan:

$$
p=P(correct|x)
$$

Expected utility:

$$
EU(a|x)
=
\sum_y P(y|x)U(a,y)
$$

Decision optimal:

$$
a^*
=
\arg\max_a EU(a|x)
$$

Kalau probability tidak calibrated, maka expected utility tersebut salah.

Artinya calibration bukan sekadar "nice metric".

Itu bisa menjadi bagian dari **decision infrastructure**.

---

# 29. Evaluation RLCD seharusnya lebih luas daripada accuracy

Kalau saya membuat benchmark RLCD, saya akan mengelompokkan eval menjadi empat lapisan.

Pertama, predictive quality:

$$
Accuracy
$$

$$
Macro-F1
$$

$$
AUROC
$$

$$
AUPRC
$$

Kedua, probabilistic quality:

$$
NLL
$$

$$
Brier
$$

$$
RPS
$$

Ketiga, calibration:

$$
ECE
$$

$$
MCE
$$

$$
Reliability\ Diagram
$$

$$
Calibration\ slope
$$

$$
Calibration\ intercept
$$

Keempat, decision quality:

$$
Risk@Coverage
$$

$$
Selective\ Accuracy
$$

$$
AURC
$$

$$
Cost\ Weighted\ Utility
$$

dan:

$$
Abstention\ Quality
$$

Contoh yang sangat bagus:

$$
Coverage=50\%
$$

kemudian:

$$
Accuracy@Coverage
$$

Artinya model hanya menerima kasus paling yakin.

Ini sangat relevan untuk automation.

---

# 30. Distribution shift harus menjadi eval wajib

Model calibrated pada training distribution:

$$
P_{train}(X,Y)
$$

belum tentu calibrated pada:

$$
P_{production}(X,Y)
$$

Misalnya:

$$
P_{train}(fraud)=0.30
$$

tetapi production:

$$
P_{prod}(fraud)=0.02
$$

Maka probability output bisa berubah maknanya.

Ini disebut antara lain prior/base-rate shift.

Oleh karena itu RLCD evaluation idealnya mencakup:

$$
in\text{-}domain
$$

$$
OOD
$$

$$
temporal\ shift
$$

$$
class\ prior\ shift
$$

$$
language\ shift
$$

$$
domain\ shift
$$

---

# 31. Calibration juga bisa gagal karena label noise

Misalkan true probability:

$$
P(Y=1|X)=0.9
$$

tetapi data Anda mengandung 20% label flipping.

Model sebenarnya akan belajar:

$$
P(Y_{observed}=1|X)
$$

bukan:

$$
P(Y_{true}=1|X)
$$

Karena itu ada prinsip yang sangat penting:

$$
\boxed{
Proper\ scoring\ rule
\ cannot\ fix\ bad\ ground\ truth
}
$$

RLCD membuat objective probabilistic menjadi lebih principled.

Ia tidak secara ajaib membuat label menjadi benar.

---

# 32. Kelebihan RLCD dibanding RLHF

Secara konseptual RLCD memiliki beberapa kelebihan yang jelas.

Tidak membutuhkan reward model besar.

RLHF:

$$
human\ preference
\rightarrow
reward\ model
\rightarrow
PPO
$$

RLCD:

$$
observed\ outcome
\rightarrow
proper\ score
\rightarrow
RL
$$

Jadi pipeline lebih sederhana.

Dan reward-nya lebih deterministic.

Tidak perlu mengatakan:

> "Jawaban A terdengar lebih baik daripada B."

Cukup:

> "Outcome aktualnya ternyata B."

Ini sangat cocok untuk domain dengan measurable outcome.

---

# 33. Tetapi RLCD mempunyai kelemahan sendiri

Pertama, ia membutuhkan outcome.

Kalau objective-nya:

> “Apa jawaban yang paling membantu manusia?”

maka outcome-nya tidak sederhana.

Contohnya:

$$
helpfulness
$$

$$
empathy
$$

$$
style
$$

$$
creativity
$$

tidak punya single ground truth natural.

Di sini human preference tetap berguna.

Kedua, calibration sangat task-dependent.

Bahkan material community sekitar Jev menunjukkan bahwa calibration perlu diverifikasi pada distribution sendiri dan tidak boleh diasumsikan sebagai universal model property. TypeSafe sendiri pada public material tidak memberikan full calibration metrics seperti ECE/Brier curve untuk Jev. ([Learn Jev][16])

Ketiga, RL mungkin tidak diperlukan jika task hanya classification sederhana.

Jika:

$$
x\rightarrow y
$$

dengan label bersih dan cross entropy bekerja baik, maka:

$$
CE
$$

bisa menjadi baseline yang sangat sulit dikalahkan.

---

# 34. Ini menurut saya research question terbesar

Pertanyaan ilmiahnya bukan:

> “Apakah RLCD lebih baik daripada RLHF?”

Karena keduanya solving problem berbeda.

Pertanyaan yang jauh lebih menarik:

> **Apakah reinforcement learning dengan proper scoring rewards memberikan improvement yang signifikan dibanding supervised cross-entropy untuk learning calibrated structured decisions?**

Kemudian:

$$
CE
$$

vs

$$
RLCD
$$

vs

$$
CE+RLCD
$$

vs

$$
CE+RLCD+KD
$$

Ini jauh lebih scientifically defensible.

---

# 35. Research design yang saya rekomendasikan

Saya akan membangun eksperimen seperti ini.

Base model:

$$
DeBERTa\text{-}v3
$$

atau bahkan:

$$
ModernBERT
$$

Dataset:

$$
Banking77
$$

$$
SNIPS
$$

$$
AGNews
$$

$$
ToxicChat
$$

$$
Enron
$$

dan satu dataset domain-specific Anda sendiri.

Kemudian empat eksperimen:

### Model A

$$
L_{CE}
$$

Baseline supervised.

### Model B

$$
L_{RLCD}
$$

pure proper scoring + RL.

### Model C

$$
L
=
L_{CE}
+
\lambda L_{RLCD}
$$

hybrid.

### Model D

$$
L
=
L_{CE}
+
\lambda L_{RLCD}
+
\mu L_{KD}
$$

hybrid + teacher distribution.

Lalu benchmark:

$$
Accuracy
$$

$$
Macro-F1
$$

$$
NLL
$$

$$
Brier
$$

$$
ECE
$$

$$
RPS
$$

$$
AURC
$$

$$
Risk@Coverage
$$

$$
Latency
$$

$$
Memory
$$

dan calibration under distribution shift.

---

# 36. Ablation study yang sangat penting

Saya akan melakukan:

$$
RLCD_{log}
$$

vs

$$
RLCD_{spherical}
$$

vs

$$
RLCD_{Brier}
$$

vs

$$
RLCD_{log+spherical}
$$

vs

$$
RLCD_{log+spherical+RPS}
$$

Kemudian:

$$
\sigma=0
$$

vs

$$
\sigma>0
$$

untuk mengukur apakah Gaussian exploration benar-benar membantu.

Kemudian:

$$
G=1
$$

vs

$$
G=4
$$

vs

$$
G=8
$$

vs

$$
G=16
$$

untuk menguji group baseline.

Dan:

$$
\lambda_{RLCD}
$$

di:

$$
0.0,\ 0.1,\ 0.25,\ 0.5,\ 1.0
$$

untuk hybrid training.

Nah, ini sudah menjadi experimental framework yang cukup kuat untuk paper.

---

# 37. Satu problem teoretis yang sangat menarik: exploration noise

Laya menggunakan Gaussian noise pada logits. Secara intuitif:

$$
z'=z+\epsilon
$$

dengan:

$$
\epsilon\sim N(0,\sigma^2)
$$

Tetapi sebenarnya:

$$
\mathbb E_\epsilon
[
S(softmax(z+\epsilon),y)
]
$$

tidak selalu mempunyai optimum yang persis sama dengan:

$$
softmax(z)=p
$$

untuk arbitrary \(\sigma\).

Jadi strictly proper scoring rule menjamin:

$$
q=p
$$

ketika Anda mengoptimalkan scoring rule terhadap true distribution secara langsung.

Tetapi ketika Anda menambahkan stochastic perturbation:

$$
q'=f(z+\epsilon)
$$

maka objective sebenarnya menjadi:

$$
J(z)
=
E_\epsilon[S(f(z+\epsilon),y)]
$$

dan optimum dapat berubah tergantung:

$$
\sigma
$$

dan bentuk scoring rule.

Ini menurut saya salah satu area riset paling menarik dalam RLCD:

$$
\boxed{
How\ does\ exploration\ noise\ affect\ calibration?
}
$$

Karena inilah bagian yang membuatnya benar-benar berbeda dari ordinary CE.

---

# 38. Kesimpulan besar dari riset ini

Setelah melihat TypeSafe, Laya, GLiNER2.5, SalesRLAgent, Qwen-2.5-1B-RLCD, dan LFM2.5-350M-RLCD, saya akan merumuskan landscape-nya seperti ini:

$$
\boxed{
System\ 1
=
Architecture
}
$$

$$
\boxed{
RLCD
=
Training\ Objective
}
$$

$$
\boxed{
REINFORCE/PPO/GRPO
=
Optimization\ Algorithm
}
$$

$$
\boxed{
Calibration
=
Statistical\ Property
}
$$

$$
\boxed{
Temperature\ Scaling
=
Post\text{-}hoc\ Calibration
}
$$

Dan yang paling penting:

$$
\boxed{
GLiNER2.5\neq RLCD
}
$$

walaupun keduanya sama-sama schema-driven encoder models.

$$
\boxed{
Qwen\text{-}RLCD\neq proof\ of\ RLCD\ training
}
$$

karena model tersebut lebih tepat dipahami sebagai parallel constrained decoding implementation.

$$
\boxed{
LFM2.5\text{-}RLCD\neq RL\ training
}
$$

karena repository-nya eksplisit menyatakan inference-only dan tanpa fine-tuning. ([Hugging Face][15])

Sementara:

$$
\boxed{
Laya
}
$$

adalah contoh open-source yang paling lengkap untuk mempelajari bagaimana ide RLCD dapat benar-benar diwujudkan dengan bidirectional encoder, probability distributions, proper scoring rewards, REINFORCE/group baseline, TD(\(\lambda\)), dan post-hoc temperature calibration. ([Hugging Face][11])

Dan:

$$
\boxed{
Jev
}
$$

adalah contoh paling menarik untuk sisi frontier/commercial, tetapi secara ilmiah harus diperlakukan sebagai **partially disclosed system**, karena TypeSafe telah mengungkap philosophy dan target objective-nya, tetapi belum membuka detail teknis yang dibutuhkan untuk mereplikasi exact algorithm. ([TypeSafe AI][1])

Kalau saya melihat ini dari perspektif riset Anda, justru ada satu thesis yang sangat kuat:

> **“Beyond Classification Accuracy: Reinforcement Learning with Proper Scoring Rules for Calibrated Structured Decisions.”**

Dengan eksperimen:

$$
CE
\quad vs \quad
RLCD
\quad vs \quad
CE+RLCD
$$

dan backbone kecil seperti:

$$
100M\text{–}400M
$$

Anda bahkan tidak perlu LLM 7B. Model seperti ModernBERT, DeBERTa, GLiNER-style encoder, atau bahkan model 100M–300M sudah cukup untuk menguji hipotesis fundamentalnya.

Itu menurut saya jauh lebih menarik secara ilmiah daripada sekadar membuat “mini Jev”, karena pertanyaannya langsung menyentuh hubungan antara **reinforcement learning, proper scoring rules, calibration, uncertainty estimation, dan non-autoregressive inference**.

[1]: https://typesafe.ai/blog/introducing-system-one-models-and-jev?utm_source=chatgpt.com "Introducing System One Models & Jev - TypeSafe AI Blog"
[2]: https://arxiv.org/html/2507.18546v1 "GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface"
[3]: https://link.springer.com/article/10.1007/s10994-023-06336-7?utm_source=chatgpt.com "Classifier calibration: a survey on how to assess and improve predicted class probabilities | Machine Learning | Springer Nature Link"
[4]: https://doi.org/10.1146%2Fannurev-statistics-042424-050626?utm_source=chatgpt.com "Proper Scoring Rules for Estimation and Forecast Evaluation | Annual Reviews"
[5]: https://proceedings.mlr.press/v235/shao24c.html?utm_source=chatgpt.com "Language Generation with Strictly Proper Scoring Rules"
[6]: https://huggingface.co/convaiinnovations/laya?utm_source=chatgpt.com "convaiinnovations/laya · Hugging Face"
[7]: https://huggingface.co/convaiinnovations/laya-multilingual/commit/4bb4d65403a3a7b8abd9e6876ccb5e75cf923b5c?utm_source=chatgpt.com "Initial release of Laya Multilingual (mmBERT-base, 100+ languages, RLCD) · convaiinnovations/laya-multilingual at 4bb4d65"
[8]: https://openai.com/index/instruction-following/?utm_source=chatgpt.com "Aligning language models to follow instructions | OpenAI"
[9]: https://arxiv.org/html/2503.23303v1 "SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization"
[10]: https://arxiv.org/abs/2503.23303 "[2503.23303] SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization"
[11]: https://huggingface.co/convaiinnovations/laya-typed-decisions?utm_source=chatgpt.com "convaiinnovations/laya-typed-decisions · Hugging Face"
[12]: https://laya.convaiinnovations.com/ "Laya — 33ms Multilingual System 1 Decision Engine with Calibrated Probabilities"
[13]: https://huggingface.co/fastino/gliner2.5-base-v1?hardware=b200&utm_source=chatgpt.com "fastino/gliner2.5-base-v1 · Hugging Face"
[14]: https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD "harshatheg/Qwen-2.5-1B-RLCD · Hugging Face"
[15]: https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD?utm_source=chatgpt.com "notnotsamuel/LFM2.5-350M-RLCD · Hugging Face"
[16]: https://learnjev.com/concepts/calibration?utm_source=chatgpt.com "Is Jev actually calibrated? · Learn Jev"
