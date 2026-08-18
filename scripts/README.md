# ReasonMotion Quantitative Evaluation and 3D Visualization Tools Guide (scripts/)

This directory contains all the helper tools for quantitative metric evaluation and 3D skeleton video generation. All scripts have built-in path fixes, so you can run `python scripts/<script_name>.py` directly from any directory in the project.

---

## 📊 1. Quantitative Metric Evaluation (Evaluation)

### 1. Base evaluation tool (`evaluate.py`)
Used to evaluate a specific checkpoint's quantitative reconstruction and physical smoothness metrics on the test set (Split 2).

*   **Usage example** (evaluate an SFT model's epoch-500 performance, averaging over 5 samples):
    ```bash
    python scripts/evaluate.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --epoch 500 \
      --nsample 5 \
      --batch_size 32
    ```
*   **Quick validation mode** (use only 1% of the dataset for a quick smoke test of the pipeline):
    ```bash
    python scripts/evaluate.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --quick
    ```

### 2. Temporal relay ablation evaluation (`evaluate_hybrid.py`)
Used to run a relay inference between an **RL model** (first-half denoising) and an **SFT model** (second-half denoising) at a specific time point $T_{\text{split}}$, verifying how denoising responsibilities are split between overall motion structure and physical jitter.

*   **Sweep metrics across different relay points (T_split)** (evaluates both the RL-first and SFT-first modes):
    ```bash
    python scripts/evaluate_hybrid.py \
      --rl_dir outputs/rl_finefs_2026-07-15_15-08 \
      --sft_dir outputs/sft_finefs_2026-07-14_00-16 \
      --t_splits 50 45 40 30 20 15 10 5 0 \
      --batch_size 32
    ```
*   **Output report**:
    After execution, a `hybrid_evaluation_report.md` Markdown report is automatically generated under the `--rl_dir` folder, containing a full table and causal-conclusion analysis of `ADE`, `FDE`, `LDLJ` (smoothness), `SPARC` (physical smoothness), and `Div_Acc` (acceleration diversity) comparing the forward and reverse relay directions.

---

## 🎥 2. 3D Skeleton Video Generation (Visualization)

### 1. All-purpose visualization detective (`visualize.py`)
Supports 3D motion rendering videos for SFT, MoE, and RL, offering 4 practical functional modes.

*   **Mode 1: `infer` (single inference comparison)**
    Generates a side-by-side comparison video of Prediction and Ground Truth (GT) for the specified motion:
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode infer \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts triple
    ```
*   **Mode 2: `grid` (checkpoint evolution grid)**
    Lays out all checkpoints horizontally and different prompts vertically, giving a panoramic view of the model's evolution over training:
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode grid \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts single double triple
    ```
*   **Mode 3: `cfg_sweep` (guidance strength sweep)**
    Sweeps different CFG scales (0.0~5.0) to observe motion diversity and the collapse point:
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode cfg_sweep \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts triple
    ```
*   **Mode 4: `diversity` (random seed comparison)**
    Fixes the prompt and generates motions simultaneously with 5 random seeds, to catch mode collapse or local bad cases:
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode diversity \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts triple
    ```

### 2. RL diversity exploration sweeper (`visualize_rl_diversity.py`)
Dedicated to sweeping, within the RL/GRPO stage's Delayed Denoising mechanism, how the denoising standard deviation and the number of active steps quantitatively and qualitatively affect generation diversity.

*   **Sweep the denoising standard deviation `sampling_std`**:
    ```bash
    python scripts/visualize_rl_diversity.py \
      --exp_dir outputs/rl_finefs_2026-07-15_15-08 \
      --sweep_type std \
      --sampling_std 0.02 0.05 0.1 0.2 0.5 1.0 \
      --num_discrete_steps 5
    ```
*   **Sweep the number of active denoising steps `num_discrete_steps`**:
    ```bash
    python scripts/visualize_rl_diversity.py \
      --exp_dir outputs/rl_finefs_2026-07-15_15-08 \
      --sweep_type steps \
      --sampling_std 1.0 \
      --num_discrete_steps 2 5 10 15 30
    ```

Videos are automatically output to `visualize_suite/` under the corresponding output directory. Diversity metrics (Mean Std, Max Std, Max Pairwise Dist) are also printed to the terminal, to quantitatively assess the degree of branch divergence.
