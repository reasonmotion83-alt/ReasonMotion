import torch
import torch.nn as nn
import os
try:
    import loralib as lora
except ImportError:
    lora = None
from typing import Optional

from utils.fs_reward_utils.modules.pose_understanding import PoseUnderstanding
from utils.fs_reward_utils.modules.make_graph import Graph

class HumanPosePerception(nn.Module):
    def __init__(self, num_class, in_channel, residual, dropout,
                 t_kernel_size, layout, strategy, hop_size, num_att_graph,
                 hpp_way, pretrain=True, lora_config=None):
        super().__init__()
        self.hpp_way = hpp_way
        self.pretrain = pretrain
        
        # Graph
        graph = Graph(layout=layout, strategy=strategy, hop_size=hop_size)
        spatial_graph = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('spatial_graph', spatial_graph)

        # config
        kwargs = dict(s_kernel_size=spatial_graph.size(0),
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=spatial_graph.size(),
                      hpp_way=self.hpp_way,
                      pretrain=self.pretrain,
                      lora_config=lora_config)

        # The setting of spatial-temporal attention graph convolutional networks (STA-GCN)
        if self.hpp_way == 'STAGCN':
            f_config = [[in_channel, 32, 1], [32, 32, 1], [32, 32, 1], [32, 64, 2], [64, 64, 1]]
            self.PoseUnderstanding = PoseUnderstanding(config=f_config, **kwargs)
            self.output_channel = f_config[-1][1]

        # "HPP" - Human Pose Perception
        else:
            understanding_config = [[in_channel, 32, 1], [32, 32, 1], [32, 64, 1], [64, 64, 1], [64, 128, 1]]
            self.PoseUnderstanding = PoseUnderstanding(config=understanding_config, **kwargs)
            self.output_channel = understanding_config[-1][1]

    def forward(self, x):
        # [N : batch size, c : channels, t : number of frame, v : joints]     
        skeleton_feature = self.PoseUnderstanding(x, self.spatial_graph)
        return skeleton_feature

class CoachScoringModel(nn.Module):
    def __init__(self, hpp_model):
        super().__init__()
        self.hpp = hpp_model
        # Simple regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hpp_model.output_channel, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        skeleton_feature = self.hpp(x)  # (B, C, T, V)
        out = self.head(skeleton_feature)  # (B, 1)
        return out.squeeze(1)  # -> (B)

class FSRewardModel(nn.Module):
    """
    Wrapper for easy loading and inference.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda", 
                 layout="SMPL_24", scale_type='linear', temperature=1.0, 
                 min_score=-5.0, max_score=5.0, **kwargs):
        super().__init__()
        self.device = device
        self.scale_type = scale_type
        self.temperature = temperature
        self.min_score = min_score
        self.max_score = max_score
        
        # Default config matching training
        hpp_backbone = HumanPosePerception(
            num_class=1024, in_channel=6, residual=True, dropout=0.5, 
            t_kernel_size=9, layout=layout, strategy='spatial', 
            hop_size=3, num_att_graph=4, hpp_way='HPP', pretrain=True, 
            lora_config=None
        )
        self.scoring_model = CoachScoringModel(hpp_backbone)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        
        self.to(device)
        self.eval()

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.scoring_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.scoring_model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            
    def forward(self, x):
        """
        Supports:
        1. (N, 6, T, V) - Already processed
        2. (N, T, V, 3) - Raw joint coordinates (needs bone calculation)
        """
        with torch.no_grad():
            if x.dim() == 4 and x.shape[-1] == 3:
                #if True: # Force print for every batch to see scale
                    #print(f"DEBUG coords: mean={x.mean().item():.4f} std={x.std().item():.4f} max={x.max().item():.4f} min={x.min().item():.4f}")
                
                # (N, T, V, 3) -> (N, 3, T, V)
                joint = x.permute(0, 3, 1, 2)
                
                # Calculate bone features
                B, C, T, V = joint.shape
                bone = torch.zeros_like(joint)
                
                # Based on SMPL_24 layout
                bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
                            (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
                
                for v1, v2 in bonelink:
                    if v2 < V:
                        bone[:, :, :, v2] = joint[:, :, :, v1] - joint[:, :, :, v2]
                
                x = torch.cat([joint, bone], dim=1) # (N, 6, T, V)

            raw_scores = self.scoring_model(x)
            #if torch.rand(1) < 0.1: # Increase frequency for debugging
                #print(f"DEBUG: raw_scores mean={raw_scores.mean().item():.4f} min={raw_scores.min().item():.4f} max={raw_scores.max().item():.4f}")

            # Scale scores to [0, 1] rewards
            if self.scale_type == 'linear':
                scores = (raw_scores - self.min_score) / (self.max_score - self.min_score + 1e-8)
                # scores = torch.clamp(scores, 0.0, 1.0) # Temporarily disabled
            elif self.scale_type == 'exp':
                normalized_scores = (raw_scores - self.min_score) / (self.max_score - self.min_score + 1e-8)
                # normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)
                scores = torch.exp(normalized_scores / self.temperature)
                # Normalize so that max_score gives 1.0 (roughly)
                scores = scores / torch.exp(torch.tensor(1.0 / self.temperature))
                # scores = torch.clamp(scores, 0.0, 1.0)
            else:
                # Default behavior
                scores = raw_scores
            
            return scores
