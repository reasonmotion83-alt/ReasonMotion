"""
FineFS Dataset loader – unified for both the DePOSit main model and the FS Reward Model.

● 資料夾結構
root/
└── train | valid | test
    └── 3A/                # 動作類別資料夾
        └── 3A_0001/
            ├── res_3A_0001.mp4
            ├── new_res.pk          # HybrIK skeleton (pred_xyz_24_struct[_global])
            └── *_info.json         # 裁判分數 (judge_score / goe)

● mode
    "full_name"     → 標籤=完整動作名稱 (e.g. "triple lutz + double toeloop combo")
    "rotation"      → 只保留單跳；標籤=圈數  (single/double/triple/quadruple)
    "combo_vs_solo" → 標籤=combo / solo

● 主模型 (DePOSit) 需要的欄位:
    pose (T, D_flat)  mask (T, D_flat)  timepoints (T,)  motion_name

● Reward 模型 (FSRewardModel) 需要的欄位:
    pose (6, T, V)  motion_name  judge_score

● 兩者統一用這一個 Dataset，各自只取自己要的 key 即可。

● random_face=True
    每次 __getitem__ 呼叫時，對整個 sliding window 沿垂直軸 (Y) 做隨機旋轉，
    消除攝影機視角造成的落地方向偏見。建議 train=True, val/test=False。
"""
from __future__ import annotations
import os, glob, pickle, json, re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------- 動作對應表 ---------- #
MOTION_NAME_MAP = {
    "A": "axel", "T": "toeloop", "S": "salchow", "Lo": "loop",
    "F": "flip", "Lz": "lutz", "Eu": "euler",
}
ROTATION_MAP = {"1": "single", "2": "double", "3": "triple", "4": "quadruple"}

# SMPL_24 Skeleton Edges (for visualization reference)
EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18),
    (18, 20), (20, 22), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
]

# Bone links for bone feature calculation (parent → child, SMPL_24 topology)
BONE_LINKS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21),
]

NUM_JOINTS = 24

# ---------- 工具函式 ---------- #
def build_mask(in_n: int, out_n: int, dim: int) -> np.ndarray:
    """簡單的 no_miss mask（主模型用）。"""
    m = np.zeros((in_n + out_n, dim), dtype=np.float32)
    m[:in_n] = 1
    return m

def expand_motion_name(code: str) -> str:
    """3Lz+2T → 'triple lutz + double toeloop combo'"""
    parts = []
    for p in code.split("+"):
        m = re.match(r"(\d)([A-Za-z]+)", p)
        if not m:
            parts.append(p); continue
        rot, base = m.groups()
        rot_name  = ROTATION_MAP.get(rot, f"{rot}-rotation")
        base_name = MOTION_NAME_MAP.get(base, base)
        parts.append(f"{rot_name} {base_name}")
    return (" + ".join(parts) + " combo") if len(parts) > 1 else (parts[0] + " solo")

def random_rotate_y(joint_data: np.ndarray) -> np.ndarray:
    """
    對骨骼序列沿垂直軸 (Y) 做隨機旋轉。

    滑冰動作的品質與落地方向無關，但攝影機視角會讓不同影片的
    落地方向集中在特別的角度，導致模型對朝向產生偏見。
    每次 __getitem__ 都隨機旋轉，迫使模型學習與方向無關的特徵。

    Args:
        joint_data: (T, V, 3) – 全域關節座標，Y 軸為垂直方向
    Returns:
        rotated (T, V, 3)
    """
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    # 繞 Y 軸旋轉矩陣 (right-hand rule):
    #  [ cos  0  sin ]
    #  [  0   1   0  ]
    #  [-sin  0  cos ]
    R = np.array([
        [ cos_t, 0.0,  sin_t],
        [ 0.0,   1.0,  0.0  ],
        [-sin_t, 0.0,  cos_t],
    ], dtype=np.float32)
    return joint_data @ R.T   # (T, V, 3) @ (3, 3)ᵀ → (T, V, 3)


# ---------- Dataset ---------- #
class FineFS(Dataset):
    def __init__(self,
                 data_dir        : str,
                 input_n         : int,
                 output_n        : int,
                 skip_rate       : int   = 1,
                 split           : int   = 0,
                 disable_sliding : bool  = False,
                 mode            : str   = "rotation",
                 data_ratio      : float = 1.0,
                 downsample      : int   = 1,
                 move_global     : bool  = True,
                 max_len         : Optional[int] = None,
                 random_face     : bool  = False,   # 隨機 Y 軸旋轉 augmentation
                 reward_mode     : bool  = False,   # True → 回傳 bone feature + judge_score
                 filter_single_rotation: bool = True):  # 過濾單圈跳（1A/1Lz 等雜訊樣本）
        """
        Args:
            random_face:  True 時每次 __getitem__ 對 sliding window 做隨機 Y 軸旋轉。
                          建議 train split 開啟，val/test 關閉。
            reward_mode:  True  → 回傳 pose=(6,T,V) tensor + judge_score (reward model 用)
                          False → 回傳 pose=(T,D) ndarray + mask + timepoints (主模型用)
        """
        super().__init__()
        assert mode in ["full_name", "rotation", "combo_vs_solo"]
        assert 0 < data_ratio <= 1

        self.mode           = mode
        self.seq_len        = input_n + output_n
        self.in_n           = input_n
        self.out_n          = output_n
        self.skip_rate      = skip_rate
        self.downsample     = downsample
        self.dim_used       = np.arange(NUM_JOINTS * 3)
        self.random_face    = random_face
        self.reward_mode    = reward_mode

        split_name = {0: "train", 1: "valid", 2: "test"}[split]
        final_dir = os.path.join(data_dir, split_name)
        if not os.path.isdir(final_dir):
            raise FileNotFoundError(f"[Error] Data folder not found: {final_dir}")
        print(f"[Init] Data folder: {final_dir}")

        # ===== 收集 pk 檔案 =====
        files = glob.glob(os.path.join(final_dir, "*", "*", "new_res.pk"))
        print(f"[{split_name}] Total pk files found: {len(files)}")

        if mode == "rotation":
            # 過濾 combo 跳（含 '+' 的都是 combo）
            files = [f for f in files if "+" not in os.path.basename(os.path.dirname(os.path.dirname(f)))]
            # 過濾單圈跳（1A/1Lz 等雜訊多）— 由 filter_single_rotation 控制
            if filter_single_rotation:
                files = [f for f in files
                         if not os.path.basename(os.path.dirname(os.path.dirname(f))).startswith("1")]
                print(f"[{split_name}] After filtering (solo, no single-rotation): {len(files)}")
            else:
                print(f"[{split_name}] After filtering (solo only, single-rotation KEPT): {len(files)}")

        files = sorted(files)
        keep_n = max(1, int(len(files) * data_ratio))
        self.pk_files = files[:keep_n]
        if data_ratio < 1.0:
            print(f"[{split_name}] data_ratio={data_ratio:.2f} → keeping {keep_n}/{len(files)} files")

        # ===== 讀取資料 & 建立滑動窗口 =====
        self.p3d         : Dict[int, np.ndarray] = {}
        self.motion_labels: Dict[int, str]       = {}
        self.judge_scores : Dict[int, float]     = {}
        self.file_paths   : Dict[int, str]       = {}   # debug 用
        self.data_idx     : List[Tuple[int, int]] = []

        total_windows = 0
        key = 0
        print(f"[{split_name}] max_len={max_len}")

        for pk in self.pk_files:
            try:
                with open(pk, "rb") as f:
                    data = pickle.load(f)
                arr = data["pred_xyz_24_struct_global" if move_global else "pred_xyz_24_struct"].astype(np.float32)
            except Exception as e:
                print(f"[Error] Failed to load {pk}: {e}")
                continue

            arr = arr[::downsample]
            if max_len is not None:
                arr = arr[:max_len]

            # 補長度
            if arr.shape[0] < self.seq_len:
                pad_len = self.seq_len - arr.shape[0]
                arr = np.concatenate([arr, np.repeat(arr[-1:], pad_len, axis=0)], 0)

            # arr shape: (T, V, 3) → flatten to (T, V*3) for storage
            arr = arr.reshape(arr.shape[0], -1)
            self.p3d[key] = arr

            # Motion label
            motion_code = os.path.basename(os.path.dirname(os.path.dirname(pk)))
            if mode == "full_name":
                label = expand_motion_name(motion_code)
            elif mode == "rotation":
                m = re.match(r"(\d)", motion_code)
                label = ROTATION_MAP.get(m.group(1), "unknown") if m else "unknown"
            else:
                label = "combo" if "+" in motion_code else "solo"
            self.motion_labels[key] = label
            self.file_paths[key]    = pk

            # Judge score（reward model 用；主模型用不到但不影響）
            pk_dir = os.path.dirname(pk)
            info_files = [f for f in os.listdir(pk_dir) if f.endswith("_info.json")]
            score_val = 0.0
            if info_files:
                try:
                    with open(os.path.join(pk_dir, info_files[0]), "r") as f:
                        info = json.load(f)
                    if "judge_score" in info:
                        j = info["judge_score"]
                        valid = [float(s) for s in (j if isinstance(j, list) else [j])
                                 if s is not None and str(s) != ""]
                        score_val = float(np.mean(valid)) if valid else 0.0
                    elif "goe" in info:
                        g = info["goe"]
                        valid = [float(v) for v in (g if isinstance(g, list) else [g])
                                 if v is not None and v != ""]
                        score_val = float(np.mean(valid)) if valid else 0.0
                except Exception as e:
                    print(f"[Warning] Failed to read score from {pk}: {e}")
            self.judge_scores[key] = score_val

            # 滑動窗口
            if disable_sliding:
                starts = [0]
            else:
                starts = list(np.arange(0, arr.shape[0] - self.seq_len + 1, skip_rate))
                #skip rate代表每隔幾幀取一個window
            if len(starts) == 0:
                starts = [0]

            self.data_idx.extend([(key, s) for s in starts])
            total_windows += len(starts)
            key += 1

        print(f"[Summary] Total sequences loaded: {key}")
        print(f"[Summary] Total sliding windows: {total_windows}")

        # 標籤分佈
        label_counts: Dict[str, int] = {}
        for k in self.motion_labels:
            lbl = self.motion_labels[k]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print("[Label Distribution]")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {lbl:<30}: {cnt}")

        # 分數分佈（reward mode でよく使う）
        scores = list(self.judge_scores.values())
        if any(s != 0.0 for s in scores):
            print(f"[Score Distribution] min={min(scores):.2f} max={max(scores):.2f} "
                  f"mean={np.mean(scores):.2f} std={np.std(scores):.2f}")

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.data_idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key, start = self.data_idx[idx]
        pose_flat = self.p3d[key][start:start + self.seq_len]   # (T, V*3)

        # Reshape to (T, V, 3) for geometry operations
        joint_data = pose_flat.reshape(pose_flat.shape[0], NUM_JOINTS, 3)   # (T, V, 3)

        # -------- random_face: 沿 Y 軸隨機旋轉 --------
        # 在 bone 計算之前做，確保 bone vector 與 joint 一致旋轉。
        # val / test 不做，保證可重複性。
        if self.random_face:
            joint_data = random_rotate_y(joint_data)
        # -----------------------------------------------

        label = self.motion_labels[key]
        score = self.judge_scores[key]

        if self.reward_mode:
            # ---- Reward Model 格式 ----
            # joint: (3, T, V),  bone: (3, T, V)  →  x: (6, T, V)
            joint = torch.tensor(joint_data, dtype=torch.float32).permute(2, 0, 1)  # (3, T, V)
            bone  = torch.zeros_like(joint)
            for v1, v2 in BONE_LINKS:
                bone[:, :, v2] = joint[:, :, v1] - joint[:, :, v2]
            x = torch.cat((joint, bone), dim=0)  # (6, T, V)
            return {
                "pose"        : x,
                "motion_name" : label,
                "judge_score" : torch.tensor(score, dtype=torch.float32),
            }
        else:
            # ---- 主模型 (DePOSit) 格式 ----
            pose_out = joint_data.reshape(self.seq_len, -1)   # (T, V*3)  ← keep as ndarray
            mask     = build_mask(self.in_n, self.out_n, pose_out.shape[1])
            return {
                "pose"        : pose_out[:, self.dim_used],
                "mask"        : mask[:, self.dim_used],
                "timepoints"  : np.arange(self.seq_len),
                "motion_name" : label,
                "judge_score" : score,  # float, 主模型通常不用
            }


# ---------------- Quick CLI test ---------------- #
if __name__ == "__main__":
    # 主模型格式
    ds = FineFS(
        data_dir    = "/home/allen/datasets/FineFS_5s/3_final",
        input_n     = 30,
        output_n    = 40,
        skip_rate   = 1,
        split       = 1,
        mode        = "rotation",
        data_ratio  = 1.0,
        max_len     = 90,
        random_face = False,
        reward_mode = False,
    )
    print("Main model sample keys:", list(ds[0].keys()))

    # Reward model 格式
    ds_r = FineFS(
        data_dir    = "/home/allen/datasets/FineFS_5s/3_final",
        input_n     = 30,
        output_n    = 40,
        split       = 0,
        mode        = "rotation",
        max_len     = 90,
        random_face = True,
        reward_mode = True,
    )
    sample = ds_r[0]
    print("Reward model sample: pose shape =", sample["pose"].shape,
          "  judge_score =", sample["judge_score"].item())