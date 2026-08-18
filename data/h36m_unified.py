from __future__ import annotations

import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import data_utils


def _resolve_h36m_base_dir(data_dir: str) -> str:
    """Resolve different H36M folder layouts used across projects."""
    candidates = [
        os.path.join(data_dir, "h3.6m/dataset"),
        os.path.join(data_dir, "h3.6m/h3.6m/dataset"),
        os.path.join(data_dir, "dataset"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "H36M folder not found. Tried: " + ", ".join(candidates)
    )


def _build_nomiss_mask(in_n: int, out_n: int, dim: int) -> np.ndarray:
    mask = np.zeros((in_n + out_n, dim), dtype=np.float32)
    mask[:in_n] = 1.0
    return mask


class H36MUnified(Dataset):
    """Unified H36M dataset for predictor/deposit compatibility.

    Key goals:
    - Keep DePOSit evaluation protocol available (find_indices_256 on split=2).
    - Keep predictor training fields (`motion_name`, `judge_score`) available.
    - Allow existing model code to run unchanged (`pose`, `mask`, `timepoints`).
    """

    def __init__(
        self,
        data_dir: str,
        input_n: int,
        output_n: int,
        skip_rate: int = 1,
        split: int = 0,
        actions: Optional[List[str]] = None,
        joints: int = 17,
        downsample: int = 2,
        max_len: Optional[int] = None,
        no_overlap: bool = False,
        protocol: str = "predictor",
        miss_type: str = "no_miss",
        miss_rate: float = 0.2,
        all_data: bool = True,
        data_ratio: float = 1.0,
        pad_short_sequences: Optional[bool] = None,
    ):
        super().__init__()
        assert split in [0, 1, 2]
        assert protocol in ["predictor", "deposit", "allen"]
        assert downsample >= 1
        assert 0 < data_ratio <= 1

        self.base_dir = _resolve_h36m_base_dir(data_dir)
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.seq_len = input_n + output_n
        self.skip_rate = skip_rate
        self.downsample = downsample
        self.max_len = max_len
        self.no_overlap = no_overlap
        self.protocol = protocol
        self.miss_type = miss_type
        self.miss_rate = miss_rate
        self.all_data = all_data
        self.data_ratio = data_ratio
        self.pad_short_sequences = (
            (protocol == "predictor") if pad_short_sequences is None else pad_short_sequences
        )
        self.pad_short_sequences = False

        if joints == 17:
            self.dim_used = np.array(
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51,
                    52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82,
                    83,
                ]
            )
        elif joints == 22:
            self.dim_used = np.array(
                [
                    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66,
                    67, 68, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91,
                    92,
                ]
            )
        elif joints == 24:
            self.dim_used = np.arange(24 * 3)
        else:
            self.dim_used = np.arange(96)

        if protocol == "deposit":
            # CoMusion Setting
            split_subjects = [[1, 5, 6, 7, 8], [11], [9, 11]]
            # AuxFormer Setting
            # split_subjects = [[1, 9, 6, 7, 8], [11], [5]]
        elif protocol == "allen":
            split_subjects = [[1, 6, 7, 8, 9], [11], [5]]
        else:
            split_subjects = [[1, 5, 6, 7, 8], [11], [9, 11]]
            # AuxFormer Setting
            # split_subjects = [[1, 6, 7, 8, 9], [11], [5]]

        self.subjects = split_subjects[split]
        self.actions = actions or [
            "walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether",
        ]

        # whole motion seqs
        self.p3d: Dict[int, np.ndarray] = {}
        self.motion_labels: Dict[int, str] = {}
        # (key, start_frame)
        self.data_idx: List[Tuple[int, int]] = []

        self._build_index()

        if self.data_ratio < 1.0:
            self._apply_data_ratio()

        print(
            f"[H36MUnified] protocol={self.protocol} split={self.split} "
            f"downsample={self.downsample} pad_short={self.pad_short_sequences} "
            f"sequences={len(self.motion_labels)} windows={len(self.data_idx)}"
        )

    def _apply_data_ratio(self):
        """Subsample windows per sequence key to preserve eval distribution."""
        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for item in self.data_idx:
            key, _start = item
            grouped.setdefault(key, []).append(item)

        rng = np.random.RandomState(1234567890)
        sampled_idx: List[Tuple[int, int]] = []
        original_count = len(self.data_idx)

        for key in sorted(grouped.keys()):
            items = grouped[key]
            keep = max(1, int(round(len(items) * self.data_ratio)))
            keep = min(keep, len(items))
            selected = rng.choice(len(items), size=keep, replace=False)
            selected.sort()
            sampled_idx.extend(items[i] for i in selected)

        self.data_idx = sampled_idx
        print(
            f"[H36MUnified] data_ratio={self.data_ratio} "
            f"kept {len(self.data_idx)}/{original_count} windows with per-sequence sampling"
        )

    def _read_xyz(self, fpath: str) -> Optional[np.ndarray]:
        if not os.path.exists(fpath):
            return None

        seq = data_utils.readCSVasFloat(fpath)
        seq = np.asarray(seq[:: self.downsample])
        if self.max_len is not None:
            seq = seq[: self.max_len]

        if seq.shape[0] < self.seq_len:
            if not self.pad_short_sequences:
                return None
            pad = self.seq_len - seq.shape[0]
            seq = np.concatenate([seq, np.repeat(seq[-1:], pad, axis=0)], axis=0)

        seq_t = torch.from_numpy(seq).float()
        seq_t[:, 0:6] = 0
        xyz = data_utils.expmap2xyz_torch(seq_t).view(seq_t.shape[0], -1).cpu().numpy()
        return xyz

    def _add_windows_regular(self, key: int, num_frames: int):
        if self.no_overlap:
            starts = [0]
        else:
            starts = np.arange(0, num_frames - self.seq_len + 1, self.skip_rate)
        self.data_idx.extend((key, int(s)) for s in starts)

    def _build_index(self):
        key = 0

        for subj in self.subjects:
            for action in self.actions:

                if self.split <= 1:
                    for subact in [1, 2]:
                        fpath = os.path.join(self.base_dir, f"S{subj}/{action}_{subact}.txt")
                        xyz = self._read_xyz(fpath)
                        if xyz is None:
                            continue

                        self.p3d[key] = xyz
                        self.motion_labels[key] = action
                        self._add_windows_regular(key, xyz.shape[0])
                        key += 1
                else:
                    if self.protocol == "deposit":
                        f1 = os.path.join(self.base_dir, f"S{subj}/{action}_1.txt")
                        f2 = os.path.join(self.base_dir, f"S{subj}/{action}_2.txt")
                        xyz1 = self._read_xyz(f1)
                        xyz2 = self._read_xyz(f2)
                        if xyz1 is None or xyz2 is None:
                            continue

                        self.p3d[key] = xyz1
                        self.motion_labels[key] = action
                        self.p3d[key + 1] = xyz2
                        self.motion_labels[key + 1] = action

                        fs_sel1, fs_sel2 = data_utils.find_indices_256(
                            xyz1.shape[0], xyz2.shape[0], self.seq_len, input_n=self.in_n
                        )
                        self.data_idx.extend((key, int(s)) for s in fs_sel1[:, 0])
                        self.data_idx.extend((key + 1, int(s)) for s in fs_sel2[:, 0])
                        key += 2
                    else:
                        for subact in [1]:
                            fpath = os.path.join(self.base_dir, f"S{subj}/{action}_{subact}.txt")
                            xyz = self._read_xyz(fpath)
                            if xyz is None:
                                continue

                            self.p3d[key] = xyz
                            self.motion_labels[key] = action
                            self._add_windows_regular(key, xyz.shape[0])
                            key += 1

    def __len__(self):
        return len(self.data_idx)

    def _build_mask(self, pose: np.ndarray) -> np.ndarray:
        if self.miss_type == "no_miss":
            return _build_nomiss_mask(self.in_n, self.out_n, pose.shape[1])

        if self.miss_type == "random_joints":
            mask = np.zeros((self.in_n, pose.shape[1]), dtype=np.float32)
            p_miss = self.miss_rate * np.ones((pose.shape[1], 1))
            for i in range(0, pose.shape[1], 3):
                A = np.random.uniform(0.0, 1.0, size=[self.in_n])
                B = A > p_miss[i]
                mask[:, i] = 1.0 * B
                mask[:, i + 1] = 1.0 * B
                mask[:, i + 2] = 1.0 * B
            predict = np.zeros((self.out_n, pose.shape[1]), dtype=np.float32)
            return np.concatenate((mask, predict), axis=0)

        # Fallback to no_miss for unsupported modes in this unified loader.
        return _build_nomiss_mask(self.in_n, self.out_n, pose.shape[1])

    def __getitem__(self, idx) -> Dict[str, Any]:
        """Returns a single sample dict, containing:
        - pose: (seq_len, len(dim_used)), in meters, keeping only the used dimensions.
        - pose_32: (seq_len, full_dim), the original 32-joint pose (millimeters).
        - mask: (seq_len, len(dim_used)), the observation mask corresponding to pose.
        - timepoints: (seq_len,) time indices [0, ..., seq_len-1].
        - motion_name: the motion name for this sequence (or "unknown" if unlabeled).
        - judge_score: default score (float, fixed at 0.0).
        """
        key, start_frame = self.data_idx[idx]
        fs = np.arange(start_frame, start_frame + self.seq_len)

        pose_32 = self.p3d[key][fs]
        pose = pose_32.copy() / 1000.0
        mask = self._build_mask(pose)

        return {
            "pose": pose[:, self.dim_used],
            "pose_32": pose_32,
            "mask": mask[:, self.dim_used],
            "timepoints": np.arange(self.seq_len),
            "motion_name": self.motion_labels.get(key, "unknown"),
            "judge_score": 0.0,
        }
