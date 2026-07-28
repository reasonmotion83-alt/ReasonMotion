"""
Boxing Multi-View Dataset loader supporting Body25 skeletons (25 joints x 3 = 75 dims).
Unified for both DePOSit / SFT main model and Boxing Reward Model.
"""

from __future__ import annotations
import os, json, random, re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.skeleton import EDGES_BODY25, BONE_LINKS_BODY25

NUM_JOINTS_BODY25 = 25

def build_mask(in_n: int, out_n: int, dim: int) -> np.ndarray:
    """CSDI mask: observed frames (first in_n) = 1, future target frames = 0."""
    m = np.zeros((in_n + out_n, dim), dtype=np.float32)
    m[:in_n] = 1
    return m


class BoxingDataset(Dataset):
    """
    Boxing Dataset Loader supporting Body25 skeletons and 5-level scoring.
    """
    def __init__(self,
                 data_dir     : str = "/home/ray/datasets/boxing_multi_view",
                 input_n      : int = 30,
                 output_n     : int = 30,

                 skip_rate    : int = 10,
                 split        : int = 0,   # 0 = train, 1 = valid, 2 = test
                 max_len      : Optional[int] = None,
                 data_ratio   : float = 1.0,
                 target_type  : str = "reactor",
                 neutral_ratio: float = 1.5,
                 reward_mode  : bool = False,
                 random_face  : bool = False):
        super().__init__()
        self.seq_len = input_n + output_n
        self.in_n = input_n
        self.out_n = output_n
        self.skip_rate = skip_rate
        self.split = split
        self.reward_mode = reward_mode
        self.random_face = random_face
        self.target_type = target_type
        self.neutral_ratio = neutral_ratio
        self.dim_used = np.arange(NUM_JOINTS_BODY25 * 3)

        train_ids = ['48kAQzqESe', 'DWpGDJidxb', 'Jkowb03BHN', 'Qv69TP2UKy', 'SXnBTmRFSK', 'WXmjVdC6hF', 'Y8Sxgb40yG', 'ujJZYtWozg']
        test_ids  = ['8e2Vh9J8Mi', '9MoIIB7Gog', 'rP8VWnNoMD', 'sz8jIrwwYv']
        
        self.video_ids = train_ids if split == 0 else test_ids

        ann_file = os.path.join(data_dir, 'annotation/complete_punch_in_round.json')
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        with open(ann_file, 'r') as f:
            database = json.load(f)['database']

        self.samples = []
        punch_samples = []
        neutral_samples = []

        split_name = ["train", "valid", "test"][split] if split < 3 else "train"
        print(f"[{split_name.upper()} BoxingDataset] Video IDs ({len(self.video_ids)}): {self.video_ids}")

        for vid in self.video_ids:
            kp_file = os.path.join(data_dir, 'EasyMocap_SMPL_keypoints', f'{vid}.json')
            if not os.path.exists(kp_file):
                print(f"[Warning] Keypoints file not found: {kp_file}")
                continue

            with open(kp_file, 'r') as f:
                kp_data = json.load(f)

            kps_dict = {}
            for pid_str, color in [('0', 'red'), ('1', 'blue')]:
                player_raw = kp_data.get(pid_str, [])
                processed = []
                last_valid = None

                for frame in player_raw:
                    if isinstance(frame, list) and len(frame) == 25:
                        valid = True
                        for joint in frame:
                            if not isinstance(joint, list) or len(joint) != 3:
                                valid = False
                                break
                        if valid:
                            kp = np.array(frame, dtype=np.float32)
                            # Center keypoints relative to pelvis (joint 8)
                            root = kp[8].copy()
                            kp = kp - root
                            processed.append(kp)
                            last_valid = kp
                            continue

                    if last_valid is not None:
                        processed.append(last_valid.copy())
                    else:
                        processed.append(np.zeros((25, 3), dtype=np.float32))

                kps_dict[color] = np.array(processed, dtype=np.float32)

            L_red = len(kps_dict['red'])
            L_blue = len(kps_dict['blue'])
            min_len = min(L_red, L_blue)

            if min_len < self.seq_len:
                continue

            red_kps = kps_dict['red'][:min_len]
            blue_kps = kps_dict['blue'][:min_len]

            red_labels = ['neutral'] * min_len
            blue_labels = ['neutral'] * min_len

            for player_color, timeline in [('red', red_labels), ('blue', blue_labels)]:
                cam_key = f"{vid}_cam1_{player_color}"
                if cam_key in database:
                    punches = database[cam_key]['annotations']
                    for punch in punches:
                        start_f, end_f = punch['segment(frames)']
                        label = punch['label']
                        for f in range(max(0, start_f), min(min_len, end_f + 1)):
                            timeline[f] = label

            starts = np.arange(0, min_len - self.seq_len + 1, skip_rate)
            for t in starts:
                red_pose_win = red_kps[t : t + self.seq_len]
                blue_pose_win = blue_kps[t : t + self.seq_len]

                red_future = red_labels[t + self.in_n : t + self.seq_len]
                blue_future = blue_labels[t + self.in_n : t + self.seq_len]

                active_red = [lbl for lbl in red_future if lbl != 'neutral']
                active_blue = [lbl for lbl in blue_future if lbl != 'neutral']

                red_action = max(set(active_red), key=active_red.count) if active_red else 'neutral'
                blue_action = max(set(active_blue), key=active_blue.count) if active_blue else 'neutral'

                s1_score = self._compute_5level_score(red_action, blue_action, red_pose_win, blue_pose_win)
                s2_score = self._compute_5level_score(blue_action, red_action, blue_pose_win, red_pose_win)

                s1 = {
                    "pose": red_pose_win,
                    "opp_pose": blue_pose_win,
                    "motion_name": red_action,
                    "opp_motion_name": blue_action,
                    "score": s1_score,
                    "is_punch": (red_action != 'neutral'),
                    "video_id": vid,
                    "start_frame": t
                }

                s2 = {
                    "pose": blue_pose_win,
                    "opp_pose": red_pose_win,
                    "motion_name": blue_action,
                    "opp_motion_name": red_action,
                    "score": s2_score,
                    "is_punch": (blue_action != 'neutral'),
                    "video_id": vid,
                    "start_frame": t
                }

                for s in [s1, s2]:
                    self.samples.append(s)

        # Apply score-level balancing for training split to prevent score=0.0 dominant bias
        if split == 0:
            score_buckets: Dict[float, List[Dict]] = {}
            for s in self.samples:
                sc = s["score"]
                score_buckets.setdefault(sc, []).append(s)

            # Find non-zero active samples count as benchmark
            active_count = sum(len(score_buckets.get(sc, [])) for sc in [2.0, 1.0, -1.0, -2.0])
            # Cap Level 3 (0.0) to ~20% of total (or ~25% of active total)
            target_zero_count = int(active_count * 0.25)


            balanced_samples = []
            for sc, samples in score_buckets.items():
                if sc == 0.0:
                    keep_n = min(len(samples), target_zero_count)
                    balanced_samples.extend(random.sample(samples, keep_n))
                else:
                    balanced_samples.extend(samples)
            
            random.shuffle(balanced_samples)
            self.samples = balanced_samples

        if data_ratio < 1.0:
            keep = int(len(self.samples) * data_ratio)
            self.samples = self.samples[:keep]

        score_counts = {}
        for s in self.samples:
            sc = s["score"]
            score_counts[sc] = score_counts.get(sc, 0) + 1

        print(f"[Summary] Loaded {len(self.samples)} samples. Score distribution: {dict(sorted(score_counts.items()))}")



    def _compute_5level_score(self, act_main: str, act_opp: str, main_pose: np.ndarray, opp_pose: np.ndarray) -> float:
        """
        攻擊動態與對手反應結合之 5 階評分 [-2.0, +2.0]：
        main_pose / opp_pose shape: (T, 25, 3)，已相對於各自 Pelvis (Joint 8) 對齊
        Body25 Keypoint 索引:
            0: Head / Nose,  1: Neck,  8: Pelvis (Root, 0,0,0)
            2: Right Shoulder, 3: Right Elbow, 4: Right Wrist
            5: Left Shoulder,  6: Left Elbow,  7: Left Wrist
        """
        is_main_punch = (act_main != 'neutral')
        is_opp_punch  = (act_opp != 'neutral')

        # 1. 計算主角擊打手 (Wrist 4/7) 相對同側肩膀 (Shoulder 2/5) 的最大手臂延伸距離 (Reach)
        r_wrist_reach = np.linalg.norm(main_pose[:, 4] - main_pose[:, 2], axis=-1)  # Right arm
        l_wrist_reach = np.linalg.norm(main_pose[:, 7] - main_pose[:, 5], axis=-1)  # Left arm
        max_reach = np.max(np.maximum(r_wrist_reach, l_wrist_reach))  # 全序列最大出拳伸展距離 (公尺)

        # 2. 計算對手頭部/軀幹 (Joint 0) 在預測視窗內的後退/避讓位移 (Opponent Disruption)
        opp_head = opp_pose[self.in_n:, 0]  # (out_n, 3)
        head_disp = np.max(np.linalg.norm(opp_head - opp_head[0], axis=-1))

        if is_main_punch and is_opp_punch:
            # Level 1 (-2.0): Countered / Vulnerable (雙方同時出拳，暴露出拳破綻)
            return -2.0  
        elif is_main_punch and not is_opp_punch:
            if max_reach < 0.38:
                # Level 2 (-1.0): Hesitant / Weak Punch (Prompt 為出拳但手臂未充分延伸，如僅深蹲防守或縮手)
                return -1.0
            elif head_disp > 0.12 and max_reach >= 0.45:
                # Level 5 (+2.0): Dominant Heavy Punch (高質量充分出拳延伸 且 迫使對手壓制後退/閃避)
                return 2.0   
            else:
                # Level 4 (+1.0): Clean Effective Punch (乾淨標準出拳延伸)
                return 1.0   
        elif not is_main_punch and is_opp_punch:
            # Level 2 (-1.0): Passive (主角未攻擊且陷入被動)
            return -1.0  
        else:
            # Level 3 (0.0): Standard Neutral (標準試探姿態)
            return 0.0



    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        joint_data = sample["pose"].copy()  # (T, 25, 3)

        if self.random_face and self.split == 0:
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
            R = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]], dtype=np.float32)
            joint_data = np.matmul(joint_data, R)

        label = sample["motion_name"]
        score = sample["score"]

        if self.reward_mode:
            joint = torch.tensor(joint_data, dtype=torch.float32).permute(2, 0, 1)  # (3, T, V)
            bone = torch.zeros_like(joint)
            for v1, v2 in BONE_LINKS_BODY25:
                if v1 < 25 and v2 < 25:
                    bone[:, :, v2] = joint[:, :, v1] - joint[:, :, v2]
            x = torch.cat((joint, bone), dim=0)  # (6, T, 25)
            return {
                "pose": x,
                "motion_name": label,
                "judge_score": torch.tensor(score, dtype=torch.float32)
            }
        else:
            pose_out = joint_data.reshape(self.seq_len, -1)  # (T, 75)
            mask = build_mask(self.in_n, self.out_n, pose_out.shape[1])
            return {
                "pose": pose_out,
                "mask": mask,
                "timepoints": np.arange(self.seq_len),
                "motion_name": label,
                "judge_score": score
            }
