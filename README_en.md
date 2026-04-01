<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/ChaoXoX/Minimind_engram?style=social)](https://github.com/ChaoXoX/Minimind_engram/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/ChaoXoX/Minimind_engram)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/ChaoXoX/Minimind_engram)](https://github.com/ChaoXoX/Minimind_engram/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/ChaoXoX/Minimind_engram/pulls)

</div>

<div align="center">
  <h3>"Knowledge in memory, reasoning in computation"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

---

# 📌 Project Introduction

**MiniMind-Engram** is an enhanced variant of the [MiniMind](https://github.com/jingyaogong/minimind) open-source architecture. Inspired by DeepSeek's latest research (MLA, MoE, and the knowledge-reasoning decoupling ideas in DeepSeek-R1), it introduces the **Engram memory module** — implementing a **knowledge-reasoning separated** language model architecture.

> "Engram" comes from the neuroscience concept of a memory engram: the set of neurons in the brain that encode a specific memory.  
> Inspired by this, every Transformer decoder layer in this project is augmented with a set of learnable key-value memory banks, allowing the model to store **objective knowledge** in dedicated parameters, decoupled from the **dynamic reasoning** pathway.

Key features:

- **Knowledge-reasoning separation**: The Engram module provides an independent knowledge retrieval path; the Transformer backbone focuses on reasoning and planning — the two work in parallel and complement each other.
- **Low VRAM overhead**: The default configuration (`engram_size=64`) adds only ~`2 × hidden_size × 64` parameters per layer, negligible compared to the FFN.
- **Plug-and-play**: Enable with `use_engram=True` in the config — no changes to any existing training pipeline.
- **Domain knowledge augmentation**: When fine-tuning on vertical domains (medical, legal, code, etc.), Engram parameters absorb domain knowledge more selectively, reducing interference with the backbone weights.

> This project inherits the full MiniMind codebase — Pretrain, SFT, LoRA, RLHF, RLAIF, and all training utilities — and extends it with the Engram module.  
> Original MiniMind project: [jingyaogong/minimind](https://github.com/jingyaogong/minimind).

---

# 📌 Engram Module Design

## Motivation: Why separate knowledge from reasoning?

Large language models typically store knowledge and reasoning capabilities in the same set of weights. Research from DeepSeek has shown that decoupling these two abilities — enabling the model to **actively retrieve** knowledge rather than passively recall it — helps improve factual accuracy, reduce hallucinations, and boost performance on knowledge-intensive tasks in specific domains.

The standard Transformer FFN layer can be interpreted as an implicit memory, but its knowledge retrieval process is deeply entangled with context-level reasoning, making it hard to adjust independently. The Engram module makes this process explicit:

```
hidden_states → [Attention] → [Engram memory retrieval] → [FFN reasoning] → output
```

## Architecture

Each `MiniMindBlock` contains three sub-modules:

```
Input hidden_states
    │
    ├─► Self-Attention (contextual reasoning)
    │       └─ residual connection
    │
    ├─► Engram Memory (objective knowledge retrieval)    ← NEW
    │       └─ residual connection
    │
    └─► FFN / MoE (linguistic patterns & compositional reasoning)
            └─ residual connection
```

### EngramMemory implementation

```python
class EngramMemory(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(hidden_size, hidden_size)          # query projection
        self.k_mem  = nn.Parameter(randn(mem_size, hidden_size))   # learnable keys
        self.v_mem  = nn.Parameter(randn(mem_size, hidden_size))   # learnable values

    def forward(self, hidden_states):
        q      = self.q_proj(hidden_states)       # (B, S, H)
        scores = q @ self.k_mem.T * scale         # (B, S, M)
        attn   = softmax(scores, dim=-1)
        return dropout(attn @ self.v_mem) * scale # (B, S, H)
```

The memory bank size `M` (`engram_size`) defaults to 64, far smaller than the FFN's `intermediate_size` (~2400), so the additional VRAM cost is minimal.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_engram` | `False` | Enable the Engram module |
| `engram_size` | `64` | Number of memory slots M |
| `engram_scale` | `1.0` | Output scaling factor |
| `engram_dropout` | same as `dropout` | Dropout rate on Engram output |

Usage:

```python
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_engram=True,     # enable Engram
    engram_size=64,      # memory slots
    engram_scale=1.0,
)
model = MiniMindForCausalLM(config)
```

---

# 📌 Quick Start

<details>
<summary>Reference hardware / software setup</summary>

* GPU: NVIDIA GeForce RTX 3090 (24 GB) × 1 or better
* Python 3.10+
* CUDA 12.2+
* Dependencies: [requirements.txt](./requirements.txt)

</details>

## Step 0: Clone & install

```bash
git clone https://github.com/ChaoXoX/Minimind_engram
cd Minimind_engram && pip install -r requirements.txt
```

## Ⅰ 🚀 Inference

### Download a model

```bash
# ModelScope
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# or HuggingFace
git clone https://huggingface.co/jingyaogong/minimind-3
```

> If you use a self-trained checkpoint with Engram enabled, make sure to pass the matching `MiniMindConfig` (`use_engram=True`) when loading.

### CLI inference

```bash
# Transformers format
python eval_llm.py --load_from ./minimind-3
# Local PyTorch weights
python eval_llm.py --load_from ./model --weight full_sft
```

### WebUI (optional)

```bash
cd scripts && streamlit run web_demo.py
```

### Third-party inference backends (optional)

```bash
# ollama
ollama run jingyaogong/minimind-3
# vllm
vllm serve /path/to/model --served-model-name "minimind"
```

## Ⅱ 🛠️ Training

### 1. Download datasets

Download from [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) or [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) and place in `./dataset/`.

Recommended for quick reproduction (✨):

```bash
./dataset/
├── pretrain_t2t_mini.jsonl  (1.2 GB ✨)
├── sft_t2t_mini.jsonl       (1.6 GB ✨)
└── rlaif.jsonl              (24 MB  ✨)
```

### 2. Pretrain

```bash
cd trainer && python train_pretrain.py
```

To train with the Engram module, set `use_engram=True` in the `MiniMindConfig` inside `train_pretrain.py`:

```python
config = MiniMindConfig(
    ...,
    use_engram=True,
    engram_size=64,
)
```

### 3. Supervised Fine-Tuning (SFT)

```bash
cd trainer && python train_full_sft.py
```

### 4. Optional training stages

| Method | Script | Description |
|--------|--------|-------------|
| LoRA | `trainer/train_lora.py` | Parameter-efficient fine-tuning |
| DPO | `trainer/train_dpo.py` | Preference learning |
| GRPO / CISPO | `trainer/train_grpo.py` | RLAIF reinforcement learning |
| PPO | `trainer/train_ppo.py` | Classic RL |
| Agentic RL | `trainer/train_agent.py` | Multi-turn tool-use RL |
| Knowledge Distillation | `trainer/train_distillation.py` | White-box distillation |

All scripts support `--from_resume 1` for checkpoint resumption and `torchrun --nproc_per_node N` for multi-GPU training.

---

# 📌 Model Architecture

## Overall structure

`MiniMind-Engram` uses a Transformer Decoder-Only architecture aligned with the `Qwen3` ecosystem:

* Pre-Norm + RMSNorm
* SwiGLU activation
* RoPE positional encoding with YaRN extrapolation support
* GQA (`q_heads=8`, `kv_heads=4`)

In each Transformer Block, an optional **Engram Memory** sub-layer is inserted between Attention and FFN, enabling the separation of knowledge and reasoning:

![structure](./images/LLM-structure.jpg)

## Parameter scale reference

| Model | Params | Engram | n_layers | d_model | kv_heads | q_heads |
|-------|--------|--------|----------|---------|----------|---------|
| minimind-3 | 64M | ✗ | 8 | 768 | 4 | 8 |
| minimind-3 + Engram | ≈64M | ✓ (M=64) | 8 | 768 | 4 | 8 |
| minimind-3-moe | 198M/A64M | ✗ | 8 | 768 | 4 | 8 |
| minimind-3-moe + Engram | ≈198M/A64M | ✓ (M=64) | 8 | 768 | 4 | 8 |

> With M=64, each Engram layer adds only `2 × 768 × 64 = 98,304` parameters. Across 8 layers this is ~786K extra parameters — roughly **1.2%** of the 64M base model — with negligible impact on VRAM.

---

# 📌 Data Formats

## Pretraining

```jsonl
{"text": "Transformer models contextual relationships through self-attention and form the backbone of modern LLMs."}
```

## SFT / Conversation

```jsonl
{
    "conversations": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]
}
```

## Tool calling

```jsonl
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "Translate 'Hello World' into Chinese"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"Hello World\",\"target_language\":\"chinese\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"你好世界\"}"},
        {"role": "assistant", "content": "你好世界"}
    ]
}
```

---

# 📌 Training Cost Reference

Single NVIDIA RTX 3090 (24 GB), `minimind-3` (64M + Engram) on `mini` datasets:

| Stage | Dataset | Time (approx.) | Cost (approx.) |
|-------|---------|----------------|----------------|
| Pretrain | pretrain_t2t_mini | ≈1.3 h | ≈1.7 ¥ |
| SFT | sft_t2t_mini | ≈1.2 h | ≈1.6 ¥ |

> The Engram module's parameter overhead is negligible; training time is essentially unchanged.

---

# 📌 Acknowledgements & Citation

This project builds on the following work — many thanks to the original authors:

- **MiniMind**: [jingyaogong/minimind](https://github.com/jingyaogong/minimind) — provides a minimal, reproducible full LLM training framework
- **DeepSeek**: The MLA, MoE, and knowledge-reasoning decoupling ideas from the DeepSeek series (DeepSeek-V2/V3/R1) heavily inspired the Engram module design
- **Memory Engram**: The neuroscience theory of memory engrams inspired the module's name and design philosophy

If you use this project, please also cite the original MiniMind:

```bibtex
@misc{minimind,
  author       = {Jingyao Gong},
  title        = {MiniMind},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jingyaogong/minimind}}
}
```

---

# 📌 License

This project is open-sourced under the [Apache 2.0](./LICENSE) license.
