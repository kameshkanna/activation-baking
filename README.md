# Norm-Calibrated Activation Baking

**Behavioural Adapters via Weight-Space Symmetry Alignment**

> ICML 2026 Workshop on Weight-Space Symmetries: from Foundations to Practical Applications

---

## Abstract

We show that activation steering vectors derived from contrastive pairs, when projected onto their PCA components, lie in **weight-space symmetry-invariant subspaces** of transformer models. We propose K = μ/√d as a principled calibration formula linked to the spectral norm of weight matrices, and validate three claims empirically across four LLM architectures: (1) PCA behavioral directions are invariant under neuron permutations, (2) they align with top singular vectors of weight matrices, and (3) K-calibrated adapters outperform uncalibrated steering. No weight updates required.

---

## Quick Start

```python
from activation_baking import Baker

baker = Baker("meta-llama/Llama-3.1-8B-Instruct", device="cuda")

baker.fit(
    positive_prompts=["..."],    # prompts eliciting desired behaviour
    negative_prompts=["..."],    # prompts eliciting undesired behaviour
    layers=(14, 28),             # layer range (start, end)
    n_components=5,              # PCA components
    k_calibration="auto",        # K = μ / √hidden (recommended)
)

output = baker.generate("What do you think about X?", alpha=1.0)
```

---

## Installation

```bash
# Clone and install
git clone <repo>
cd norms_k_analysis
pip install -e .

# Or just install dependencies
pip install -r requirements.txt
```

---

## Models

| Architecture | Model | Hidden | Layers | VRAM |
|---|---|---|---|---|
| Llama | `meta-llama/Llama-3.1-8B-Instruct` | 4096 | 32 | ~16 GB |
| Qwen | `Qwen/Qwen2.5-7B-Instruct` | 3584 | 28 | ~14 GB |
| Gemma | `google/gemma-2-9b-it` | 3584 | 42 | ~18 GB |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | 4096 | 32 | ~14 GB |

All fit on a single **A10G (24 GB)** or **A100 (40 GB)**.

---

## Experiment Pipeline

Run scripts in order. Each is standalone and saves results to `results/`.

```bash
# 1. Profile activation norms per architecture (~30 min on A100)
python experiments/01_norm_profiling.py --model all --device cuda

# 2. Extract PCA behavioral directions from contrastive pairs (~45 min)
python experiments/02_contrastive_extraction.py --model all --behavior all --device cuda

# 3. Validate K-formula vs. spectral norms (~30 min)
python experiments/03_k_calibration_validation.py --model all --device cuda

# 4. KEY: Permutation invariance experiment (~60 min)
python experiments/04_permutation_invariance.py --model all --behavior all --device cuda

# 5. Baking efficacy comparison (~90 min)
python experiments/05_baking_efficacy.py --model all --behavior all --device cuda

# 6. KEY: Weight-space alignment (~45 min)
python experiments/06_weight_space_alignment.py --model all --behavior all --device cuda

# 7. Cross-architecture CKA (~60 min)
python experiments/07_cross_arch_comparison.py --device cuda
```

**Total GPU time: ~7 hours. Cost on Lambda Labs A100: ~$10.**

---

## Generate Paper Figures & Tables

```bash
python analysis/plotting.py --results-dir results --output-dir results/plots
python analysis/tables.py   --results-dir results --output-dir results/plots
```

Outputs saved as `.pdf` (300 DPI) and `.png` to `results/plots/`.

---

## Results Structure

```
results/
├── norm_profiles/           # 01: per-layer mean norm + K-values
├── pca_directions/          # 02: fitted PCA directions per model × behavior
├── k_calibration/           # 03: K vs spectral norm correlation
├── permutation_invariance/  # 04: cosine sim before/after permutation  ← KEY
├── efficacy/                # 05: method comparison accuracy scores
├── weight_alignment/        # 06: PCA alignment with weight SVD        ← KEY
├── cross_arch/              # 07: CKA cross-architecture similarity
└── plots/                   # paper figures (PDF + PNG) + LaTeX tables
```

---

## Behaviors (Contrastive Pair Datasets)

| Behavior | Pairs | Description |
|---|---|---|
| `sycophancy_suppression` | 60 | Honest disagreement vs. validation of false claims |
| `refusal_calibration` | 60 | Answering benign borderline questions vs. over-refusing |
| `verbosity_control` | 50 | Concise answers vs. verbose responses |
| `formality` | 50 | Formal academic register vs. casual/colloquial |
| `uncertainty_expression` | 50 | Calibrated hedging vs. overconfident claims |

---

## Repository Structure

```
norms_k_analysis/
├── activation_baking/          # Core package (pip-installable)
│   ├── model_utils.py          # Architecture detection, neuron permutation
│   ├── extractor.py            # Hook-based activation extraction
│   ├── calibrator.py           # K = μ/√d, spectral norm computation
│   ├── pca_director.py         # PCA on contrastive diffs → behavioral axes
│   ├── baker.py                # Main user API (fit / generate / evaluate)
│   ├── evaluator.py            # Behavioral shift metrics
│   └── __init__.py
├── experiments/
│   ├── 01_norm_profiling.py
│   ├── 02_contrastive_extraction.py
│   ├── 03_k_calibration_validation.py
│   ├── 04_permutation_invariance.py   # ← Proves permutation invariance
│   ├── 05_baking_efficacy.py
│   ├── 06_weight_space_alignment.py   # ← Proves weight-space alignment
│   └── 07_cross_arch_comparison.py
├── data/behaviors/             # 5 JSONL contrastive pair datasets
├── config/
│   ├── models.yml              # 4 model definitions
│   └── experiments.yml         # Hyperparameters for all experiments
├── analysis/
│   ├── plotting.py             # 6 paper figures
│   └── tables.py               # 3 LaTeX tables
├── results/                    # Auto-populated by experiments
├── paper/
│   ├── main.tex                # 4-page ICML workshop paper
│   └── references.bib
├── setup.py
└── requirements.txt
```

---

## Lambda Labs Deployment

```bash
# Spin up: 1x A100 SXM4 40GB  ($1.50/hr)
# SSH in, then:

git clone <repo> && cd norms_k_analysis
pip install -r requirements.txt

# Login to HuggingFace (for Llama — needs license accepted at hf.co/meta-llama)
huggingface-cli login

# Run full pipeline (~7 hours)
python experiments/01_norm_profiling.py --model all --device cuda
python experiments/02_contrastive_extraction.py --model all --behavior all --device cuda
python experiments/03_k_calibration_validation.py --model all --device cuda
python experiments/04_permutation_invariance.py --model all --behavior all --device cuda
python experiments/05_baking_efficacy.py --model all --behavior all --device cuda
python experiments/06_weight_space_alignment.py --model all --behavior all --device cuda
python experiments/07_cross_arch_comparison.py --device cuda

# Generate paper outputs
python analysis/plotting.py && python analysis/tables.py
```

To run models in parallel across 2 GPUs (A100 80GB or 2x A10G):
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/01_norm_profiling.py --model llama --device cuda &
CUDA_VISIBLE_DEVICES=1 python experiments/01_norm_profiling.py --model qwen  --device cuda &
wait
```

---

## Paper

**Title:** Norm-Calibrated Activation Baking: Behavioural Adapters via Weight-Space Symmetry Alignment

**Venue:** ICML 2026 Workshop on Weight-Space Symmetries: from Foundations to Practical Applications

**Submission deadline:** April 30, 2026 (AOE) — [OpenReview](https://openreview.net/group?id=ICML.cc/2026/Workshop/WSS)

**Format:** 4 pages

---

## License

MIT
