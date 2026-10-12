nohup: ignoring input
+ /usr/bin/python3 -m pip install -q unsloth datasets
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
== BUILD (sekali, dipakai before+after) ==
==((====))==  Unsloth 2026.10.3: Fast Lfm2 patching. Transformers: 5.17.0.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu130. CUDA: 7.5. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading weights:   0%|          | 0/148 [00:00<?, ?it/s]Loading weights:   1%|          | 1/148 [00:00<00:23,  6.14it/s]Loading weights:   8%|▊         | 12/148 [00:00<00:02, 54.23it/s]Loading weights:  24%|██▎       | 35/148 [00:00<00:00, 122.42it/s]Loading weights:  41%|████      | 61/148 [00:00<00:00, 171.81it/s]Loading weights:  59%|█████▉    | 88/148 [00:00<00:00, 203.79it/s]Loading weights:  76%|███████▋  | 113/148 [00:00<00:00, 217.52it/s]Loading weights:  99%|█████████▊| 146/148 [00:00<00:00, 247.49it/s]Loading weights: 100%|██████████| 148/148 [00:00<00:00, 189.75it/s]
Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]Fetching 6 files: 100%|██████████| 6/6 [00:00<00:00, 1170.61it/s]
typed_decisions v0: 400 items (skipped 0)
banking77 v0: 500 items (skipped 0)
clinc150 v0: 500 items (skipped 0)
== BEFORE (base + head random) ==
`causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
before typed_decisions: (0.3055, 400)
before banking77: (0.024, 500)
before clinc150: (0.172, 500)
== AFTER (final latih) ==
==((====))==  Unsloth 2026.10.3: Fast Lfm2 patching. Transformers: 5.17.0.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu130. CUDA: 7.5. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading weights:   0%|          | 0/148 [00:00<?, ?it/s]Loading weights:   1%|          | 1/148 [00:00<00:24,  6.01it/s]Loading weights:  17%|█▋        | 25/148 [00:00<00:01, 113.44it/s]Loading weights:  36%|███▌      | 53/148 [00:00<00:00, 178.60it/s]Loading weights:  54%|█████▍    | 80/148 [00:00<00:00, 209.88it/s]Loading weights:  73%|███████▎  | 108/148 [00:00<00:00, 229.77it/s]Loading weights:  90%|████████▉ | 133/148 [00:00<00:00, 231.58it/s]Loading weights: 100%|██████████| 148/148 [00:00<00:00, 204.07it/s]
Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]Fetching 6 files: 100%|██████████| 6/6 [00:00<00:00, 1330.61it/s]
after typed_decisions: (0.5555, 400) 56.0ms/item
after banking77: (0.838, 500) 125.5ms/item
after clinc150: (0.738, 500) 221.3ms/item
== CI (bootstrap AFTER) ==
ci typed_decisions: [n/a]
ci banking77: [n/a]
ci clinc150: [n/a]
{'holdout_internal': 0.7902439024390244, 'n': 136}
{'calibrate_full': {'accuracy': 0.8, 'ece': 0.07645931687110506, 'loss': 0.5926404013977763, 'record_accuracy': 0.75, 'fitted_types': [0, 1, 2]}, 'n': 136}

| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |
| LFM2.5-350M-decs1 | 31% -> **56%** [n/a] | 2% -> **84%** [n/a] | 17% -> **74%** [n/a] | 79% (train-split seed=42 aug=full loto=none train/cal/test=28870/136/136) |
saved eval_results.json — tempel ke docs/RUN_LOG.md
done
