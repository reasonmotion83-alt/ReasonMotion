# utils/h36m.py
"""
資料夾結構
root/                       ← data_dir
└── h3.6m/dataset/
    └── S1/
        ├── walking_1.txt
        └── ...
"""

from __future__ import annotations
import os, glob, math, random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import data_utils    

# ------------- 共用工具 ------------- #
def build_mask(in_n: int, out_n: int, dim: int) -> np.ndarray:
    """與 FineFS 相同：前 in_n 幀 = 1，其餘 0。"""
    m = np.zeros((in_n + out_n, dim), dtype=np.float32)
    m[:in_n] = 1
    return m

# ------------- Dataset ------------- #
class H36M(Dataset):
    """
    Args
    ----
    data_dir   : str,   e.g. "/home/allen/datasets"
    input_n    : int,   長度同 FineFS
    output_n   : int
    skip_rate  : int,   sliding window stride (ignored if no_overlap=True)
    split      : int,   0=train, 1=valid, 2=test
    actions    : list[str] or None
    joints     : int,   24 / 22 / 17 / 32
    data_ratio : 0~1,   只取部分資料做快速實驗
    downsample : int,   原始序列下采樣 (預設 2)
    max_len    : int or None, 序列長度上限
    no_overlap : bool,  若 True，每個影片只用一次 (從頭開始)；若 False，使用 sliding window
    """
    def __init__(self,
                 data_dir     : str,
                 input_n      : int,
                 output_n     : int,
                 skip_rate    : int = 1,
                 split        : int = 0,
                 actions      : Optional[List[str]] = None,
                 joints       : int = 17,
                 data_ratio   : float = 1,
                 downsample   : int = 2,
                 max_len      : Optional[int] = None,
                 no_overlap   : bool = False):
        super().__init__()
        assert 0 < data_ratio <= 1
        assert split in [0, 1, 2]

        # ---------- 基本設定 ---------- #
        self.seq_len   = input_n + output_n
        self.in_n      = input_n
        self.out_n     = output_n
        self.skip_rate = skip_rate
        self.downsample= downsample
        self.split     = split
        self.no_overlap= no_overlap

        # ---------- 關節維度挑選 ---------- #
        if joints == 24:          # 與 FineFS 對齊 (HybrIK-24 skeleton)
            self.dim_used = np.arange(24 * 3)
        elif joints == 22:        # 與 DePOSit 原版一致
            self.dim_used = np.array(
                [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66,
                 67, 68, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91,
                 92])
        elif joints == 17:        # 常見 17-joint subset
            self.dim_used = np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23,
                 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51,
                 52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82,
                 83])
        else:                     # 32 關節全保留
            self.dim_used = np.arange(96)

        # ---------- subjects split ---------- #
        subs   = [[1, 5, 6, 7, 8], [11], [9, 11]][split]
        acts   = actions or ["walking", "eating", "smoking", "discussion",
                             "directions", "greeting", "phoning", "posing",
                             "purchases", "sitting", "sittingdown",
                             "takingphoto", "waiting", "walkingdog",
                             "walkingtogether"]

        # ---------- 路徑 ---------- #
        base_dir = os.path.join(data_dir, "h3.6m/h3.6m/dataset")
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"H36M folder not found: {base_dir}")
        print(f"[Init] H36M base: {base_dir}")
        print("downsample:", downsample)
        print("out_n:", self.out_n)
        # ----------- 讀取序列 ----------- #
        self.p3d: Dict[int, np.ndarray] = {}
        self.motion_labels: Dict[int, str] = {}
        self.data_idx: List[Tuple[int, int]] = []

        key        = 0
        total_win  = 0
        sample_max = None  # 若要限制樣本數，可自行設定

        for subj in subs:
            for act in acts:
                for subact in ([1, 2] if split <= 1 else [1]):  # test split 讀雙檔
                    fname = f"S{subj}/{act}_{subact}.txt"
                    fpath = os.path.join(base_dir, fname)
                    if not os.path.exists(fpath):
                        continue

                    # -------------- expmap → xyz -------------- #
                    seq = data_utils.readCSVasFloat(fpath)
                    seq = np.array(seq[::downsample])
                    if max_len is not None:
                        seq = seq[:max_len]
                    if seq.shape[0] < self.seq_len:
                        pad = self.seq_len - seq.shape[0]
                        seq = np.concatenate([seq, np.repeat(seq[-1:], pad, axis=0)], 0)

                    seq = torch.from_numpy(seq).float()
                    seq[:, 0:6] = 0           # 清除 global rot / trans
                    xyz = data_utils.expmap2xyz_torch(seq).view(seq.shape[0], -1).cpu().numpy()

                    self.p3d[key] = xyz
                    self.motion_labels[key] = act   # ← label = action name

                    # ---------- sliding windows ---------- #
                    if no_overlap:
                        starts = [0]  # 每個影片只取開頭
                    else:
                        starts = np.arange(0, xyz.shape[0] - self.seq_len + 1, skip_rate)
                    self.data_idx.extend([(key, s) for s in starts])
                    key += 1
                    total_win += len(starts)

                    if sample_max is not None and len(self.data_idx) >= sample_max:
                        break
            if sample_max is not None and len(self.data_idx) >= sample_max:
                break

        # ---------- sub-sample for data_ratio ---------- #
        if data_ratio < 1.0:
            keep = int(len(self.data_idx) * data_ratio)
            self.data_idx = self.data_idx[:keep]
            print(f"[DataRatio] keep {keep}/{total_win} sliding windows")

        print(f"[Summary] sequences: {len(self.motion_labels)} | windows: {len(self.data_idx)}")

    # ---------- 標準函式 ---------- #
    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx) -> Dict[str, Any]:
        key, start = self.data_idx[idx]
        pose = self.p3d[key][start: start + self.seq_len] / 1000.0         # (T, D)
        mask = build_mask(self.in_n, self.out_n, pose.shape[1])     # (T, D)

        return {
            "pose"       : pose[:, self.dim_used],                  # (T, used_D)
            "mask"       : mask[:, self.dim_used],
            "timepoints" : np.arange(self.seq_len),
            "motion_name": self.motion_labels[key],                 # e.g. "walking"
            "judge_score": 0.0,                                     # 佔位
        }
# ------------- Skeleton Edge (for H36M) ------------- #
# 32 關節版本 (full)；22/17 只會畫到已有的頂點 index
EDGES_H36M = [
    (0, 1), (1, 2), (2, 3),        # 右腿：hip → rhip → rknee → rfoot
    (0, 4), (4, 5), (5, 6),        # 左腿：hip → lhip → lknee → lfoot
    (0, 7), (7, 8), (8, 9), (9,10),# spine → thorax → neck → head
    (8,11), (11,12), (12,13),     # 左手：thorax → lshoulder → lelbow → lwrist
    (8,14), (14,15), (15,16)      # 右手：thorax → rshoulder → relbow → rwrist
]


# ------------- 可視化函式 ------------- #
def visualize_sequence(pose: np.ndarray,
                       edges : List[Tuple[int,int]],
                       save_path: Optional[str] = None,
                       pause: float = 0.001):
    """
    pose : (T, J*3) numpy
    edges: list of (i,j)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    T, D = pose.shape
    J = D // 3
    xyz = pose.reshape(T, J, 3)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    lines = [ax.plot([], [], [], lw=2)[0] for _ in edges]
    scat  = ax.scatter([], [], [], s=15, c='r')

    def update_frame(t):
        pts = xyz[t]
        pts = pts[:, [0, 2, 1]]  # Swap Y and Z

        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for k, (i, j) in enumerate(edges):
            if i >= J or j >= J:         # 22/17 joints 情況
                lines[k].set_data([], [])
                lines[k].set_3d_properties([])
                continue
            xs, ys, zs = pts[[i, j]].T
            lines[k].set_data(xs, ys)
            lines[k].set_3d_properties(zs)
        plt.pause(pause)

    if save_path:
        import matplotlib.animation as animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='H36M'), bitrate=1800)
        ani = animation.FuncAnimation(fig, update_frame, frames=T, interval=30)
        ani.save(save_path, writer=writer)
    else:
        for t in range(T):
            update_frame(t)
        plt.show()

# ---------------- CLI quick test & visualization ---------------- #
if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/allen/datasets/h3.6m")
    parser.add_argument("--vis", type=int, default=None, help="index to visualize")
    parser.add_argument("--save", type=str, default=None, help="output mp4 path")
    args = parser.parse_args()

    ds = H36M(
        data_dir   = args.data_dir,
        input_n    = 25,
        output_n   = 100,
        skip_rate  = 1,
        split      = 0,
        joints     = 17,
        data_ratio = 1.0
    )
    print("len :", len(ds))
    sample = ds[0]

    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"{k:<12}: shape = {v.shape}")
        else:
            print(f"{k:<12}: {v}")

    if args.vis is not None:
        sample = ds[args.vis]
        print(f"▶ Visualizing idx={args.vis} | motion='{sample['motion_name']}'")
        visualize_sequence(sample["pose"], EDGES_H36M, save_path=args.save)
