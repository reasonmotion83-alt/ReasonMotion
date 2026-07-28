# ReasonMotion 定量評估與 3D 視覺化工具使用指南 (scripts/)

本目錄收錄所有用於定量指標評估與 3D 骨架影片生成的輔助工具。所有的腳本均已內嵌路徑修正，你可以在專案任何目錄下直接使用 `python scripts/<script_name>.py` 執行。

---

## 📊 一、 定量指標評估 (Evaluation)

### 1. 基礎評估工具 (`evaluate.py`)
用於在測試集 (Split 2) 上評估特定 Checkpoint 的定量重建與物理平滑指標。

*   **使用範例** (評估 SFT 模型第 500 代的表現，採樣 5 次求平均):
    ```bash
    python scripts/evaluate.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --epoch 500 \
      --nsample 5 \
      --batch_size 32
    ```
*   **快速驗證模式** (僅用 1% 的資料集快篩跑通流程):
    ```bash
    python scripts/evaluate.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --quick
    ```

### 2. 時序接力消融評估 (`evaluate_hybrid.py`)
用於將 **RL 模型**（前半去噪）與 **SFT 模型**（後半去噪）在特定時間點 $T_{\text{split}}$ 進行接力推理，驗證動作大結構與物理抖動的去噪特徵分工。

*   **掃描不同接力點 (T_split) 指標** (同時評估 RL-first 與 SFT-first 兩種模式):
    ```bash
    python scripts/evaluate_hybrid.py \
      --rl_dir outputs/rl_finefs_2026-07-15_15-08 \
      --sft_dir outputs/sft_finefs_2026-07-14_00-16 \
      --t_splits 50 45 40 30 20 15 10 5 0 \
      --batch_size 32
    ```
*   **輸出報告**:
    執行完成後，會自動在 `--rl_dir` 資料夾下生成 `hybrid_evaluation_report.md` Markdown 報告，包含正反向接力對比的 `ADE`、`FDE`、`LDLJ` (平滑度)、`SPARC` (物理平滑) 與 `Div_Acc` (加速度多樣性) 的完整表格與因果結論分析。

---

## 🎥 二、 3D 骨架影片生成 (Visualization)

### 1. 全能視覺化神探 (`visualize.py`)
支持 SFT、MoE 與 RL 的 3D 動作渲染影片，提供 4 種實用功能模式。

*   **模式 1: `infer` (單一推理對比)**
    生成指定動作的 Prediction 與 Ground Truth (GT) 雙排對比影片：
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode infer \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts triple
    ```
*   **模式 2: `grid` (Checkpoint 進化網格)**
    將所有 Checkpoints 橫向排列，縱向為不同 Prompts，展現模型隨訓練進化的全景圖：
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode grid \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts single double triple
    ```
*   **模式 3: `cfg_sweep` (引導強度掃描)**
    掃描不同 CFG Scale (0.0~5.0) 下的動作多樣性與崩潰點：
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode cfg_sweep \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts triple
    ```
*   **模式 4: `diversity` (隨機種子對比)**
    固定 Prompt，使用 5 種隨機種子同時生成動作，用以捕捉 Mode Collapse 或局部壞點：
    ```bash
    python scripts/visualize.py \
      --exp_dir outputs/sft_finefs_2026-07-14_00-16 \
      --mode diversity \
      --motion ./data/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk \
      --texts triple
    ```

### 2. RL 多樣性探索掃描器 (`visualize_rl_diversity.py`)
專門用於掃描 RL/GRPO 階段的 Delayed Denoising 機制中，去噪標準差與主動步數對生成多樣性的定量與定性影響。

*   **掃描去噪標準差 `sampling_std`**:
    ```bash
    python scripts/visualize_rl_diversity.py \
      --exp_dir outputs/rl_finefs_2026-07-15_15-08 \
      --sweep_type std \
      --sampling_std 0.02 0.05 0.1 0.2 0.5 1.0 \
      --num_discrete_steps 5
    ```
*   **掃描主動去噪步數 `num_discrete_steps`**:
    ```bash
    python scripts/visualize_rl_diversity.py \
      --exp_dir outputs/rl_finefs_2026-07-15_15-08 \
      --sweep_type steps \
      --sampling_std 1.0 \
      --num_discrete_steps 2 5 10 15 30
    ```

影片將自動輸出在對應輸出目錄的 `visualize_suite/` 中。同時會在終端機打印多樣性度量標準（Mean Std, Max Std, Max Pairwise Dist），用以定量判斷分岔擴散的程度。
