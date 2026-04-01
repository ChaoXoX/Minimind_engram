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
  <h3>"知识在记忆中，推理在计算中"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

---

# 📌 项目介绍

**MiniMind-Engram** 是基于 [MiniMind](https://github.com/jingyaogong/minimind) 开源架构的增强版本，结合 DeepSeek 系列最新研究（MLA、MoE、DeepSeek-R1 知识-推理分离思想），引入了 **Engram 记忆模块**，实现了一套**知识与推理分离**的增强型语言模型架构。

> "Engram"源自神经科学中的记忆痕迹（memory engram）概念，代表大脑中存储特定记忆的神经元集合。  
> 受此启发，本项目为 Transformer 的每个解码器层附加一组可学习的键值记忆库，使模型能够将**客观知识**固化在独立参数中，并与**动态推理**路径解耦。

核心特性：

- **知识-推理分离**：Engram 模块提供独立的知识检索路径，Transformer 主干专注于推理与规划，两者并行互补。
- **低显存开销**：默认配置（`engram_size=64`）仅在每层增加约 `2 × hidden_size × 64` 个参数，相比 FFN 层可忽略不计。
- **即插即用**：通过配置 `use_engram=True` 即可激活，不改变任何现有训练流程。
- **领域知识增强**：在垂直领域场景（医疗、法律、代码等）微调时，Engram 参数能够更集中地吸收领域知识，减少对主干权重的干扰。

> 本项目继承自 MiniMind，完整保留了其预训练、SFT、LoRA、RLHF、RLAIF 等全流程代码，并在此基础上扩展了 Engram 模块。
> 原始 MiniMind 项目详见：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)。

---

# 📌 Engram 模块设计

## 动机：为什么需要知识与推理分离？

大型语言模型通常将知识与推理能力混合存储在同一套权重中。DeepSeek 系列研究表明，将这两种能力解耦——让模型在推理时能够**主动检索**而非被动回忆知识——有助于提升事实性、减少幻觉，并改善特定领域的知识密集型任务表现。

标准 Transformer FFN 层可以被理解为一种隐式记忆，但其知识检索过程与上下文推理深度耦合，难以独立调节。Engram 模块将这一过程显式化：

```
hidden_states → [Attention] → [Engram 记忆检索] → [FFN 推理] → output
```

## 架构

每个 `MiniMindBlock` 包含三个子模块：

```
输入 hidden_states
    │
    ├─► Self-Attention（上下文推理）
    │       └─ residual connection
    │
    ├─► Engram Memory（客观知识检索）      ← 新增模块
    │       └─ residual connection
    │
    └─► FFN / MoE（语言规律与组合推理）
            └─ residual connection
```

### EngramMemory 实现

```python
class EngramMemory(nn.Module):
    def __init__(self, config):
        self.q_proj  = nn.Linear(hidden_size, hidden_size)   # 查询投影
        self.k_mem   = nn.Parameter(randn(mem_size, hidden_size))  # 可学习键
        self.v_mem   = nn.Parameter(randn(mem_size, hidden_size))  # 可学习值

    def forward(self, hidden_states):
        q      = self.q_proj(hidden_states)          # (B, S, H)
        scores = q @ self.k_mem.T * scale            # (B, S, M)
        attn   = softmax(scores, dim=-1)
        return dropout(attn @ self.v_mem) * scale    # (B, S, H)
```

记忆库大小 `M`（`engram_size`）默认为 64，远小于 FFN 的 intermediate_size（≈2400），因此额外显存开销极小。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_engram` | `False` | 是否启用 Engram 模块 |
| `engram_size` | `64` | 记忆库槽位数量 M |
| `engram_scale` | `1.0` | Engram 输出缩放系数 |
| `engram_dropout` | 同 `dropout` | Engram 输出 Dropout 率 |

启用方式：

```python
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_engram=True,      # 启用 Engram
    engram_size=64,       # 记忆槽位数
    engram_scale=1.0,
)
model = MiniMindForCausalLM(config)
```

---

# 📌 快速开始

<details>
<summary>参考软硬件配置</summary>

* GPU: NVIDIA GeForce RTX 3090 (24GB) × 1 或更高
* Python == 3.10+
* CUDA == 12.2+
* 依赖见 [requirements.txt](./requirements.txt)

</details>

## 第0步：克隆仓库、安装依赖

```bash
git clone https://github.com/ChaoXoX/Minimind_engram
cd Minimind_engram && pip install -r requirements.txt
```

## Ⅰ 🚀 推理

### 下载模型

```bash
# ModelScope
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# 或 HuggingFace
git clone https://huggingface.co/jingyaogong/minimind-3
```

> 若使用启用了 Engram 的自训练权重，确保加载时传入对应的 `MiniMindConfig`（`use_engram=True`）。

### CLI 推理

```bash
# 加载 Transformers 格式模型
python eval_llm.py --load_from ./minimind-3
# 加载本地 PyTorch 权重
python eval_llm.py --load_from ./model --weight full_sft
```

### WebUI（可选）

```bash
cd scripts && streamlit run web_demo.py
```

### 第三方推理框架（可选）

```bash
# ollama
ollama run jingyaogong/minimind-3
# vllm
vllm serve /path/to/model --served-model-name "minimind"
```

## Ⅱ 🛠️ 训练

### 1' 下载数据集

从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 下载所需数据文件，放入 `./dataset/` 目录。

快速复现推荐下载（✨）：

```bash
./dataset/
├── pretrain_t2t_mini.jsonl  (1.2GB ✨)
├── sft_t2t_mini.jsonl       (1.6GB ✨)
└── rlaif.jsonl              (24MB  ✨)
```

### 2' 预训练

```bash
cd trainer && python train_pretrain.py
```

若要以 Engram 模式训练，修改 `train_pretrain.py` 中的 `MiniMindConfig`：

```python
config = MiniMindConfig(
    ...,
    use_engram=True,
    engram_size=64,
)
```

### 3' 监督微调（SFT）

```bash
cd trainer && python train_full_sft.py
```

### 4' 其它训练（可选）

| 训练方式 | 脚本 | 说明 |
|----------|------|------|
| LoRA 微调 | `trainer/train_lora.py` | 轻量参数高效微调 |
| DPO | `trainer/train_dpo.py` | 偏好学习 |
| GRPO / CISPO | `trainer/train_grpo.py` | RLAIF 强化学习 |
| PPO | `trainer/train_ppo.py` | 经典强化学习 |
| Agentic RL | `trainer/train_agent.py` | 多轮工具调用 RL |
| 知识蒸馏 | `trainer/train_distillation.py` | 白盒蒸馏 |

所有脚本支持 `--from_resume 1` 断点续训，以及 `torchrun --nproc_per_node N` 多卡训练。

---

# 📌 模型架构

## 整体结构

`MiniMind-Engram` 基于 Transformer Decoder-Only 架构，结构配置对齐 `Qwen3` 生态：

* Pre-Norm + RMSNorm
* SwiGLU 激活函数
* RoPE 旋转位置编码，支持 YaRN 外推
* GQA（`q_heads=8`，`kv_heads=4`）

在每个 Transformer Block 中，Attention 之后、FFN 之前，新增 **Engram Memory** 子层（可选），实现知识与推理的分离：

![structure](./images/LLM-structure.jpg)

## 参数规模参考

| Model | Params | Engram | n_layers | d_model | kv_heads | q_heads |
|-------|--------|--------|----------|---------|----------|---------|
| minimind-3 | 64M | ✗ | 8 | 768 | 4 | 8 |
| minimind-3 + Engram | ≈64M | ✓ (M=64) | 8 | 768 | 4 | 8 |
| minimind-3-moe | 198M/A64M | ✗ | 8 | 768 | 4 | 8 |
| minimind-3-moe + Engram | ≈198M/A64M | ✓ (M=64) | 8 | 768 | 4 | 8 |

> Engram 模块（M=64）每层仅新增约 `2 × 768 × 64 = 98,304` 个参数，8 层共约 `786K`，相对于 64M 基础模型增量约 **1.2%**，对显存无显著影响。

---

# 📌 数据格式

## 预训练数据

```jsonl
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

## SFT / 对话数据

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮您？"}
    ]
}
```

## 工具调用数据

```jsonl
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成英文"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

---

# 📌 训练开销参考

基于单卡 NVIDIA RTX 3090（24GB），`minimind-3`（64M + Engram）在 `mini` 数据集上：

| 阶段 | 数据集 | 时间（约） | 成本（约） |
|------|--------|------------|------------|
| 预训练 | pretrain_t2t_mini | ≈1.3h | ≈1.7¥ |
| SFT | sft_t2t_mini | ≈1.2h | ≈1.6¥ |

> Engram 模块增加的参数量极小，训练时间几乎不受影响。

---

# 📌 致谢与引用

本项目基于以下工作，致谢原作者的贡献：

- **MiniMind**：[jingyaogong/minimind](https://github.com/jingyaogong/minimind) — 提供了极简可复现的 LLM 完整训练框架
- **DeepSeek**：DeepSeek 系列（DeepSeek-V2/V3/R1）的 MLA、MoE 与知识-推理解耦思想对本项目设计有重要启发
- **Memory Engram**：神经科学中记忆痕迹（memory engram）理论，启发了本模块的命名与设计哲学

如使用本项目，请同时引用原始 MiniMind 项目：

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

# 📌 许可证

本项目沿用 [Apache 2.0](./LICENSE) 协议开源。
