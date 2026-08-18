"""
FineFS Dataset loader – unified for both the DePOSit main model and the FS Reward Model.

● Folder structure
root/
└── train | valid | test
    └── 3A/                # motion category folder
        └── 3A_0001/
            ├── res_3A_0001.mp4
            ├── new_res.pk          # HybrIK skeleton (pred_xyz_24_struct[_global])
            └── *_info.json         # judge score (judge_score / goe)

● mode
    "full_name"     → label = full motion name (e.g. "triple lutz + double toeloop combo")
    "rotation"      → keep single jumps only; label = number of rotations (single/double/triple/quadruple)
    "combo_vs_solo" → label = combo / solo

● Fields required by the main model (DePOSit):
    pose (T, D_flat)  mask (T, D_flat)  timepoints (T,)  motion_name

● Fields required by the Reward model (FSRewardModel):
    pose (6, T, V)  motion_name  judge_score

● Both use this single unified Dataset; each just picks out the keys it needs.

● random_face=True
    On every __getitem__ call, applies a random rotation about the vertical axis (Y)
    to the whole sliding window, removing the landing-direction bias introduced by
    camera viewpoint. Recommended: train=True, val/test=False.
"""
from __future__ import annotations
import os, glob, pickle, json, re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------- Motion name lookup table ---------- #
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

# ---------- Utility functions ---------- #
def build_mask(in_n: int, out_n: int, dim: int) -> np.ndarray:
    """Simple no_miss mask (used by the main model)."""
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
    Applies a random rotation about the vertical axis (Y) to a skeleton sequence.

    Skating motion quality is independent of landing direction, but camera
    viewpoint causes the landing direction across different videos to cluster
    around particular angles, biasing the model toward orientation. Rotating
    randomly on every __getitem__ call forces the model to learn
    orientation-independent features.

    Args:
        joint_data: (T, V, 3) – global joint coordinates, with Y as the vertical axis
    Returns:
        rotated (T, V, 3)
    """
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    # Rotation matrix about the Y axis (right-hand rule):
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
                 random_face     : bool  = False,   # random Y-axis rotation augmentation
                 reward_mode     : bool  = False,   # True → return bone feature + judge_score
                 filter_single_rotation: bool = True):  # filter out single-rotation jumps (1A/1Lz etc., noisy samples)
        """
        Args:
            random_face:  When True, each __getitem__ call applies a random Y-axis
                          rotation to the sliding window. Recommended on for the
                          train split, off for val/test.
            reward_mode:  True  → return pose=(6,T,V) tensor + judge_score (for the reward model)
                          False → return pose=(T,D) ndarray + mask + timepoints (for the main model)
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

        # ===== Collect pk files =====
        files = glob.glob(os.path.join(final_dir, "*", "*", "new_res.pk"))
        print(f"[{split_name}] Total pk files found: {len(files)}")

        if mode == "rotation":
            # Filter out combo jumps (anything containing '+' is a combo)
            files = [f for f in files if "+" not in os.path.basename(os.path.dirname(os.path.dirname(f)))]
            # Filter out single-rotation jumps (1A/1Lz etc. tend to be noisy) — controlled by filter_single_rotation
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

        # ===== Load data & build sliding windows =====
        self.p3d         : Dict[int, np.ndarray] = {}
        self.motion_labels: Dict[int, str]       = {}
        self.judge_scores : Dict[int, float]     = {}
        self.file_paths   : Dict[int, str]       = {}   # for debugging
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

            # Pad length
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

            # Judge score (used by the reward model; unused by the main model but harmless)
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

            # Sliding window
            if disable_sliding:
                starts = [0]
            else:
                starts = list(np.arange(0, arr.shape[0] - self.seq_len + 1, skip_rate))
                # skip_rate is the frame stride between successive windows
            if len(starts) == 0:
                starts = [0]

            self.data_idx.extend([(key, s) for s in starts])
            total_windows += len(starts)
            key += 1

        print(f"[Summary] Total sequences loaded: {key}")
        print(f"[Summary] Total sliding windows: {total_windows}")

        # Label distribution
        label_counts: Dict[str, int] = {}
        for k in self.motion_labels:
            lbl = self.motion_labels[k]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print("[Label Distribution]")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {lbl:<30}: {cnt}")

        # Score distribution (commonly used in reward mode)
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

        # -------- random_face: random rotation about the Y axis --------
        # Done before the bone computation, to keep bone vectors and joints
        # rotated consistently. Skipped for val/test to ensure reproducibility.
        if self.random_face:
            joint_data = random_rotate_y(joint_data)
        # -----------------------------------------------

        label = self.motion_labels[key]
        score = self.judge_scores[key]

        if self.reward_mode:
            # ---- Reward Model format ----
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
            # ---- Main model (DePOSit) format ----
            pose_out = joint_data.reshape(self.seq_len, -1)   # (T, V*3)  ← keep as ndarray
            mask     = build_mask(self.in_n, self.out_n, pose_out.shape[1])
            return {
                "pose"        : pose_out[:, self.dim_used],
                "mask"        : mask[:, self.dim_used],
                "timepoints"  : np.arange(self.seq_len),
                "motion_name" : label,
                "judge_score" : score,  # float, usually unused by the main model
            }


# ---------------- Quick CLI test ---------------- #
if __name__ == "__main__":
    # Main model format
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

    # Reward model format
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