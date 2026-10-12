+ /usr/bin/python3 -m pip install -q unsloth datasets
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 82.4/82.4 kB 4.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28.4/28.4 MB 74.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.1/43.1 MB 19.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 73.8/73.8 kB 6.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 55.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 44.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 53.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 56.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.6/3.6 MB 58.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.9/216.9 kB 24.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.4/3.4 MB 58.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 230.5/230.5 kB 22.9 MB/s eta 0:00:00
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
== BUILD (sekali, dipakai before+after) ==
==((====))==  Unsloth 2026.10.3: Fast Lfm2 patching. Transformers: 5.17.0.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu130. CUDA: 7.5. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
/usr/local/lib/python3.13/dist-packages/huggingface_hub/constants.py:302: FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.
  warnings.warn(
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/148 [00:00<?, ?it/s]Loading weights:   1%|          | 1/148 [00:00<00:23,  6.22it/s]Loading weights:   3%|▎         | 4/148 [00:00<00:09, 15.37it/s]Loading weights:  19%|█▉        | 28/148 [00:00<00:01, 98.43it/s]Loading weights:  38%|███▊      | 56/148 [00:00<00:00, 160.89it/s]Loading weights:  55%|█████▌    | 82/148 [00:00<00:00, 192.19it/s]Loading weights:  73%|███████▎  | 108/148 [00:00<00:00, 212.83it/s]Loading weights:  89%|████████▉ | 132/148 [00:00<00:00, 220.52it/s]Loading weights: 100%|██████████| 148/148 [00:00<00:00, 176.33it/s]
Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]Fetching 6 files: 100%|██████████| 6/6 [00:00<00:00, 1271.84it/s]
Generating train split:   0%|          | 0/1200 [00:00<?, ? examples/s]Generating train split: 100%|██████████| 1200/1200 [00:00<00:00, 29422.87 examples/s]
Generating test split:   0%|          | 0/400 [00:00<?, ? examples/s]Generating test split: 100%|██████████| 400/400 [00:00<00:00, 76291.28 examples/s]
typed_decisions v0: 400 items (skipped 0)
Generating train split:   0%|          | 0/10003 [00:00<?, ? examples/s]Generating train split: 100%|██████████| 10003/10003 [00:00<00:00, 986100.62 examples/s]
Generating test split:   0%|          | 0/3080 [00:00<?, ? examples/s]Generating test split: 100%|██████████| 3080/3080 [00:00<00:00, 673889.22 examples/s]
banking77 v0: 500 items (skipped 0)
Generating train split:   0%|          | 0/15250 [00:00<?, ? examples/s]Generating train split: 100%|██████████| 15250/15250 [00:00<00:00, 748229.37 examples/s]
Generating validation split:   0%|          | 0/3100 [00:00<?, ? examples/s]Generating validation split: 100%|██████████| 3100/3100 [00:00<00:00, 430014.30 examples/s]
Generating test split:   0%|          | 0/5500 [00:00<?, ? examples/s]Generating test split: 100%|██████████| 5500/5500 [00:00<00:00, 560150.35 examples/s]
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
Loading weights:   0%|          | 0/148 [00:00<?, ?it/s]Loading weights:   1%|          | 1/148 [00:00<00:26,  5.58it/s]Loading weights:  18%|█▊        | 26/148 [00:00<00:01, 113.63it/s]Loading weights:  36%|███▌      | 53/148 [00:00<00:00, 172.66it/s]Loading weights:  54%|█████▍    | 80/148 [00:00<00:00, 206.11it/s]Loading weights:  73%|███████▎  | 108/148 [00:00<00:00, 229.82it/s]Loading weights:  93%|█████████▎| 137/148 [00:00<00:00, 248.86it/s]Loading weights: 100%|██████████| 148/148 [00:00<00:00, 204.59it/s]
Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]Fetching 6 files: 100%|██████████| 6/6 [00:00<00:00, 1302.51it/s]
['build_dataset', 'calibrate', 'evaluate', 'for_inference', 'for_training', 'freeze_backbone', 'from_pretrained', 'get_peft_model', 'predict', 'split_holdout', 'unfreeze_backbone']
    @staticmethod
    def evaluate(
        model,
        tokenizer,
        items: list,
        batch_size = None,
    ) -> dict:
        logits, items = _decision_logits(model, tokenizer, items, batch_size)
        return _metrics(logits, items, _served_temperatures(model.decision_config, logits, items))

after typed_decisions: (0.5755, 400) 56.1ms/item
typed_decisions v1: 400 items (skipped 0)
typed_decisions v2: 400 items (skipped 0)
para typed_decisions: median 57.6% range [57.6%, 57.6%]
after banking77: (0.822, 500) 124.2ms/item
banking77 v1: 500 items (skipped 0)
banking77 v2: 500 items (skipped 0)
para banking77: median 82.2% range [81.8%, 82.6%]
after clinc150: (0.556, 500) 220.2ms/item
clinc150 v1: 500 items (skipped 0)
clinc150 v2: 500 items (skipped 0)
para clinc150: median 56.0% range [55.6%, 57.0%]
== CI (bootstrap AFTER) ==
ci typed_decisions: [n/a]
ci banking77: [n/a]
ci clinc150: [n/a]
{'holdout_internal': 0.79, 'n': 139}
{'calibrate_full': {'accuracy': 0.78, 'ece': 0.049098969474434886, 'loss': 0.6765519135532463, 'record_accuracy': 0.7194244604316546, 'fitted_types': [0, 1, 2]}, 'n': 139}

| Model | typed-decisions | BANKING77 | CLINC150 | Holdout acc |
| LFM2.5-350M-decs1 | 31% -> **58%** [n/a] | 2% -> **82%** [n/a] | 17% -> **56%** [n/a] | 79% (train-split seed=3407 aug=full loto=clinc150 train/cal/test=28913/139/139) |
saved eval_results.json — tempel ke docs/RUN_LOG.md
done
