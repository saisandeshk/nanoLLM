nanoLLM/
├── configs/
│   ├── model/
│   │   ├── nanoLLM_0.6B.yaml
│   │   └── nanoLLM_1.2B.yaml (TODO)
│   ├── data/
│   │   └── smollm.yaml
│   ├── train/
│   │   ├── default.yaml
+   │   ├── fsdp_bf16.yaml           # <-- TODO: Config for a specific distributed strategy
+   │   └── ddp_h200.yaml            # <-- TODO: Example of another training config
│   └── config.yaml
├── data/
│   └── tokenizer.json
+   └── smolLM/                 # <-- TODO: Example of a pre-tokenized, memory-mapped dataset
+       ├── train.bin
+       └── val.bin
├── docs/
│   └── project_notes.md
│   └── contributions.md
│   └── use.md
├── notebooks/                  # <-- TODO
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_component_test.ipynb
│   ├── 03_pre_training.ipynb
│   └── 04_inference.ipynb
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_training.py
│   └── run_generation.py
+   └── run_evaluation.py          # <-- TODO: Script to run model evals (e.g., lm-eval-harness)
├── src/
│   └── nanoLLM/
│       ├── __init__.py
│       ├── architecture/
│       │   ├── __init__.py
│       │   ├── attention.py
│       │   ├── block.py
│       │   ├── ffn.py
│       │   ├── model.py
│       │   └── rope.py
│       │   └── embedding.py
│       ├── moe/
│       │   ├── __init__.py
│       │   ├── moe_layer.py
│       │   ├── router.py
│       │   ├── loss.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── collator.py
-       │   ├── dataset.py             # Can be kept, but MMapDataset is better for scale
+       │   ├── memmap_dataset.py      # <-- TODO: Dataset for pre-tokenized, memory-mapped data
│       │   └── tokenizer.py
│       │   └── utils.py
│       ├── distributed/
│       │   ├── __init__.py
-       │   ├── checkpoint.py          # Will be moved into a more generic training checkpoint module
│       │   └── utils.py               # DDP/FSDP setup, get_rank, world_size etc.
+       │   └── fsdp.py                # <-- TODO: FSDP-specific wrapping policies and helpers
+       │   └── tp.py                  # <-- TODO: (Optional) Tensor Parallelism utilities
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py             # The main training loop orchestrator
+       │   ├── callbacks.py           # <-- TODO: For things like logging, eval, checkpointing
+       │   ├── state.py               # <-- TODO: A dataclass to hold training state (step, epoch, etc.)
│       │   └── optimizer.py           # Optimizer creation (e.g., AdamW with fused/8-bit options)
│       │   └── lr_scheduler.py
+       │   └── checkpoint.py          # <-- REVISED: A more robust, distributed-aware checkpoint manager
│       ├── kernels/
│       │   ├── __init__.py
│       │   ├── flash_attn.py
+       │   └── fused_norm.py          # <-- TODO: Example for other fused kernels (e.g. RMSNorm)
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── generate.py
+       │   ├── kv_cache.py            # <-- TODO: Explicit KV Cache management for efficient generation
+       │   └── quantization.py        # <-- TODO: Utilities for AWQ, GPTQ, or bitsandbytes
+       ├── evaluation/              # <-- TODO: Module for calculating metrics
+       │   ├── __init__.py
+       │   └── perplexity.py
│       └── utils/
│           ├── __init__.py
-           ├── config.py              # This is good, but could be merged into src/nanoLLM/config.py
│           ├── logging.py
+           └── tracking.py            # <-- TODO: Wrappers for W&B, TensorBoard, etc.
+   ├── config.py                      # <-- REVISED: Central Pydantic config models for the whole project
├── tests/
│   ├── architecture/
+   │   └── test_model_forward.py
+   │   └── test_attention.py
│   └── data/
+   │   └── test_dataset.py
+   └── training/
+       └── test_checkpoint.py
├── pyproject.toml
└── README.md