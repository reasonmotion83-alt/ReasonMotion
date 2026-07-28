"""
NOTE: Please use the MetricsEvaluator class for evaluation instead of using the helper functions (ampjpe, fmpjpe) directly.

Usage:
    from utils.metrics import MetricsEvaluator

    evaluator = MetricsEvaluator(config, device='cuda')
    metrics = evaluator.evaluate(model, dataloader, nsample=1)
"""
import torch
import numpy as np
from tqdm import tqdm
from scipy import fft
from scipy.spatial.distance import pdist

from utils.text_encoder import TextEncoder


def compute_mean_acceleration(positions):
    """
    Compute mean acceleration magnitude across all joints and time steps.
    Lower value indicates less jitter / smoother motion.
    
    Args:
        positions: Motion positions array of shape (T, J, 3) or (T, J*3)
    Returns:
        mean_acc: Mean acceleration magnitude float
    """
    positions = np.asarray(positions)
    if positions.ndim == 2:
        T, K = positions.shape
        positions = positions.reshape(T, K // 3, 3)
    
    if positions.shape[0] < 3:
        return 0.0
    
    vel = np.diff(positions, axis=0)
    acc = np.diff(vel, axis=0)
    acc_mag = np.linalg.norm(acc, axis=-1)
    return float(np.mean(acc_mag))


def compute_ldlj(positions, fps=25):
    """
    Compute Log Dimensionless Jerk (LDLJ) for motion smoothness.
    
    LDLJ = -ln|DLJ|
    
    where DLJ = -((t2-t1)^5 / v_peak^2) * integral(|d^2v/dt^2|^2 dt)
    
    Args:
        positions: Motion positions array of shape (T, J, 3) or (T, J*3)
                   where T is time steps, J is number of joints
        fps: Frames per second of the motion data
    
    Returns:
        ldlj: Log Dimensionless Jerk value (higher/less negative = smoother)
    
    Reference:
        Balasubramanian et al. "On the analysis of movement smoothness" 
        Journal of NeuroEngineering and Rehabilitation (2015)
    """
    positions = np.asarray(positions)
    
    # Reshape to (T, J, 3) if needed
    if positions.ndim == 2:
        T, K = positions.shape
        positions = positions.reshape(T, K // 3, 3)
    
    T = positions.shape[0]
    dt = 1.0 / fps
    duration = (T - 1) * dt
    
    if T < 4:
        return 0.0  # Not enough frames to compute jerk
    
    # Compute velocity: v(t) = dx/dt
    velocity = np.diff(positions, axis=0) / dt  # (T-1, J, 3)
    
    # Compute speed (magnitude of velocity) for each joint
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, J)
    
    # Average speed across all joints
    avg_speed = speed.mean(axis=1)  # (T-1,)
    
    # Peak speed
    v_peak = avg_speed.max()
    
    if v_peak < 1e-10:
        return 0.0  # No movement
    
    # Compute acceleration: a(t) = dv/dt (second derivative)
    acceleration = np.diff(avg_speed) / dt  # (T-2,)
    
    # Compute jerk: j(t) = da/dt (third derivative)  
    jerk = np.diff(acceleration) / dt  # (T-3,)
    
    # Compute integral of |jerk|^2 using trapezoidal rule
    jerk_squared_integral = np.trapz(jerk ** 2, dx=dt)
    
    # Dimensionless Jerk (DLJ)
    dlj = -((duration ** 5) / (v_peak ** 2)) * jerk_squared_integral
    
    # Log Dimensionless Jerk (LDLJ)
    if abs(dlj) < 1e-10:
        return 0.0
    
    ldlj = -np.log(abs(dlj))
    
    return ldlj


def compute_sparc(positions, fps=30, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Compute SPectral ARC length (SPARC) for motion smoothness.
    
    SPARC = -integral_0^wc sqrt((1/wc)^2 + (dV_hat/dw)^2) dw
    
    where V_hat(w) = V(w) / V(0) is the normalized Fourier magnitude spectrum
    
    Args:
        positions: Motion positions array of shape (T, J, 3) or (T, J*3)
                   where T is time steps, J is number of joints
        fps: Frames per second of the motion data
        padlevel: Zero padding level for FFT (power of 2 multiplier)
        fc: Maximum cutoff frequency in Hz (default 10 Hz for human motion)
        amp_th: Amplitude threshold for adaptive cutoff (default 0.05)
    
    Returns:
        sparc: SPARC value (higher/less negative = smoother)
                Typical values: ~-1.6 for healthy movements, more negative for impaired
    
    Reference:
        Balasubramanian et al. "On the analysis of movement smoothness" 
        Journal of NeuroEngineering and Rehabilitation (2015)
    """
    positions = np.asarray(positions)
    
    # Reshape to (T, J, 3) if needed
    if positions.ndim == 2:
        T, K = positions.shape
        positions = positions.reshape(T, K // 3, 3)
    
    T = positions.shape[0]
    dt = 1.0 / fps
    
    if T < 4:
        return 0.0  # Not enough frames
    
    # Compute velocity: v(t) = dx/dt
    velocity = np.diff(positions, axis=0) / dt  # (T-1, J, 3)
    
    # Compute speed (magnitude of velocity) for each joint
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, J)
    
    # Average speed across all joints
    avg_speed = speed.mean(axis=1)  # (T-1,)
    
    if avg_speed.max() < 1e-10:
        return 0.0  # No movement
    
    # Zero padding for better frequency resolution
    nfft = int(2 ** (np.ceil(np.log2(len(avg_speed))) + padlevel))
    
    # Compute FFT
    speed_spectrum = np.abs(fft.fft(avg_speed, n=nfft))
    
    # Frequency axis
    freq = fft.fftfreq(nfft, d=dt)
    
    # Only positive frequencies up to fc
    pos_freq_mask = (freq >= 0) & (freq <= fc)
    freq_pos = freq[pos_freq_mask]
    spectrum_pos = speed_spectrum[pos_freq_mask]
    
    # Normalize spectrum
    if spectrum_pos[0] < 1e-10:
        return 0.0
    
    spectrum_norm = spectrum_pos / spectrum_pos[0]
    
    # Adaptive cutoff: find where normalized spectrum drops below threshold
    below_th = np.where(spectrum_norm < amp_th)[0]
    if len(below_th) > 0:
        cutoff_idx = below_th[0]
    else:
        cutoff_idx = len(freq_pos) - 1
    
    # Ensure at least some frequency content
    cutoff_idx = max(cutoff_idx, 2)
    
    freq_sel = freq_pos[:cutoff_idx + 1]
    spectrum_sel = spectrum_norm[:cutoff_idx + 1]
    
    if len(freq_sel) < 2:
        return 0.0
    
    # Compute arc length: integral of sqrt((1/wc)^2 + (dV_hat/dw)^2)
    # Use numerical differentiation for dV_hat/dw
    dfreq = np.diff(freq_sel)
    dspectrum = np.diff(spectrum_sel)
    
    # Avoid division by zero
    dfreq = np.where(dfreq < 1e-10, 1e-10, dfreq)
    
    # Derivative dV_hat/dw
    d_spectrum_d_freq = dspectrum / dfreq
    
    # Cutoff frequency
    wc = freq_sel[-1] if freq_sel[-1] > 0 else 1.0
    
    # Arc length integrand: sqrt((1/wc)^2 + (dV_hat/dw)^2)
    integrand = np.sqrt((1.0 / wc) ** 2 + d_spectrum_d_freq ** 2)
    
    # Compute arc length using trapezoidal integration
    arc_length = np.trapz(integrand, freq_sel[:-1])
    
    # SPARC is negative arc length
    sparc = -arc_length
    
    return sparc

def compute_diversity(pred, *args):
    pred = np.asarray(pred)
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item() if dist.size > 0 else 0.0
    return diversity


def get_diversity(activation, diversity_times):
    """Implementation of average Euclidean distance of M pairs."""
    num_samples = activation.shape[0]
    if num_samples < 2: return 0.0
    
    # Generate M unique unordered pairs (i, j) where i < j
    if num_samples < 200:
        indices = np.array([(i, j) for i in range(num_samples) for j in range(i + 1, num_samples)])
    else:
        # Randomly sample pairs for efficiency
        indices = np.random.randint(0, num_samples, size=(diversity_times * 2, 2))
        indices = indices[indices[:, 0] < indices[:, 1]] # Ensure i < j and unique

    if len(indices) > diversity_times:
        indices = indices[np.random.choice(len(indices), diversity_times, replace=False)]
    
    # Calculate ||z_i - z_j||_2
    diff = activation[indices[:, 0]] - activation[indices[:, 1]]
    return np.linalg.norm(diff, axis=1).mean()


def compute_ade(pred, gt, *args):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def ampjpe(batch_pred, batch_gt, target_dim):
    """Calculates A-MPJPE by reshaping (B, N, target_dim, T) to (B, N, T, J, 3).
    Computes best-of-N error: min error among N samples for each batch item.
    """
    B, N, K, T = batch_pred.shape
    p = batch_pred.transpose(-1, -2).reshape(B, N, T, target_dim // 3, 3)
    g = batch_gt.transpose(-1, -2).reshape(B, N, T, target_dim // 3, 3)

    # Calculate Euclidean distance: (B, N, T, J)
    dist = torch.norm(g - p, p=2, dim=-1)

    # Average over time (T) and joints (J) to get error per sample: (B, N)
    error_per_sample = dist.mean(dim=(-1, -2))

    # Best of N: min error over samples -> (B,)
    best_error = error_per_sample.min(dim=1)[0]

    # Average over batch
    return best_error.mean() * 1000


def fmpjpe(batch_pred, batch_gt, target_dim):
    """Calculates F-MPJPE for the final frame of the sequence.
    Computes best-of-N error: min error among N samples for each batch item.
    """
    B, N, K, T = batch_pred.shape
    p = batch_pred[..., -1].reshape(B, N, target_dim // 3, 3)
    g = batch_gt[..., -1].reshape(B, N, target_dim // 3, 3)

    # Calculate Euclidean distance: (B, N, J)
    dist = torch.norm(g - p, p=2, dim=-1)

    # Average over joints (J) to get error per sample: (B, N)
    error_per_sample = dist.mean(dim=-1)

    # Best of N: min error over samples -> (B,)
    best_error = error_per_sample.min(dim=1)[0]

    # Average over batch
    return best_error.mean() * 1000


def root_relative_without_root(batch_pose, root_idx=0):
    """Convert (B, N, K, T) pose to root-relative joints and drop root joint."""
    B, N, K, T = batch_pose.shape
    if K % 3 != 0:
        raise ValueError(f"Pose dimension K must be divisible by 3, got {K}")

    joint_count = K // 3
    if joint_count <= 1:
        raise ValueError("Need at least 2 joints to remove root and evaluate relative pose.")
    if root_idx < 0 or root_idx >= joint_count:
        raise ValueError(f"root_idx {root_idx} is out of range for {joint_count} joints.")

    pose = batch_pose.transpose(-1, -2).reshape(B, N, T, joint_count, 3)
    root = pose[..., root_idx:root_idx + 1, :]
    rel_pose = pose - root

    keep_joints = torch.ones(joint_count, dtype=torch.bool, device=batch_pose.device)
    keep_joints[root_idx] = False
    rel_pose = rel_pose[..., keep_joints, :]
    return rel_pose.reshape(B, N, T, (joint_count - 1) * 3).transpose(-1, -2).contiguous()


class MetricsEvaluator:
    def __init__(self, config, device='cuda', target_dim=None):
        self.device = device
        model_cfg = config.get('model', {})
        if target_dim is not None:
            self.target_dim = target_dim
        else:
            self.target_dim = model_cfg.get('target_dim', 72)
        
        data_cfg = config.get('dataset', config.get('data', {}))
        self.input_n = data_cfg.get('input_n', 0)
        self.root_idx = data_cfg.get('root_joint_idx', data_cfg.get('root_idx', 0))
        data_name = data_cfg.get('name', 'FineFS')
        if data_name is None:
            data_name = 'FineFS'
        data_name = str(data_name).lower()
        if data_name == 'h36m':
            default_fps = 25
        elif data_name == 'boxing':
            default_fps = 60
        else:
            default_fps = 30
        self.fps = data_cfg.get('fps', default_fps)
        self.text_encoder = TextEncoder(device=str(device))


    def evaluate(self, model, dataloader, nsample=1):
        """
        Evaluate the model on a given dataloader.

        Args:
            model: The PyTorch model to evaluate.
            dataloader: DataLoader providing the validation dataset.
            nsample (int): Number of samples to generate per input. (if > 1, the result is best-of-N)

        Returns:
            dict: A dictionary containing the computed metrics:
                - "AMPJPE"/"FMPJPE": Absolute A/F-MPJPE in mm.
                - "ADE"/"FDE": Absolute GSPS-style best-of-N displacement error.
                - "MMADE"/"MMFDE": Absolute multi-modal metrics, when nsample > 1.
                - "Diversity": Absolute mean pairwise distance among generated samples.
                - "LDLJ": Log Dimensionless Jerk (smoothness metric).
                - "SPARC": Spectral Arc Length (smoothness metric).
                - "Div_Spatial": Spatial diversity.
                - "Div_Vel": Velocity diversity.
                - "Div_Acc": Acceleration diversity.
                - "R_*": Root-relative versions of the same metrics, with root joint removed.
        """
        model.eval()
        amp_total, fmp_total = 0.0, 0.0
        ade_total, fde_total = 0.0, 0.0
        mmade_total, mmfde_total = 0.0, 0.0
        diversity_total = 0.0
        r_amp_total, r_fmp_total = 0.0, 0.0
        r_ade_total, r_fde_total = 0.0, 0.0
        r_mmade_total, r_mmfde_total = 0.0, 0.0
        r_diversity_total = 0.0
        ldlj_total, sparc_total = 0.0, 0.0
        mean_acc_total = 0.0
        samples_list = []
        n_sequences = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                t_emb = self.text_encoder(batch.get("motion_name")) if self.text_encoder else None
                
                try:
                    samples, gt = model.evaluate(batch, nsample, t_emb)[:2] # (B, N, K, T)
                    if gt.dim() == 3: gt = gt.unsqueeze(1)
                    gt = gt.expand_as(samples)

                    # Only evaluate predicted frames (input_n:)
                    p_abs, g_abs = samples[..., self.input_n:], gt[..., self.input_n:]
                    p_rel = root_relative_without_root(p_abs, root_idx=self.root_idx)
                    g_rel = root_relative_without_root(g_abs, root_idx=self.root_idx)

                    # Batch-level A/F-MPJPE (mm)
                    batch_size = p_abs.shape[0]
                    amp_total += float(ampjpe(p_abs, g_abs, p_abs.shape[2]).item()) * batch_size
                    fmp_total += float(fmpjpe(p_abs, g_abs, p_abs.shape[2]).item()) * batch_size
                    r_amp_total += float(ampjpe(p_rel, g_rel, p_rel.shape[2]).item()) * batch_size
                    r_fmp_total += float(fmpjpe(p_rel, g_rel, p_rel.shape[2]).item()) * batch_size

                    pred_abs_np = p_abs.cpu().numpy().transpose(0, 1, 3, 2)  # (B, N, T, K)
                    gt_abs_np = g_abs.cpu().numpy().transpose(0, 1, 3, 2)     # (B, N, T, K)
                    pred_rel_np = p_rel.cpu().numpy().transpose(0, 1, 3, 2)
                    gt_rel_np = g_rel.cpu().numpy().transpose(0, 1, 3, 2)

                    # Store samples for diversity calculation
                    samples_list.append(p_abs.cpu().numpy())

                    for b in range(pred_abs_np.shape[0]):
                        pred_b = pred_abs_np[b]   # (N, T, K)
                        gt_b = gt_abs_np[b, 0]    # (T, K)
                        pred_rel_b = pred_rel_np[b]
                        gt_rel_b = gt_rel_np[b, 0]

                        ade_total += float(compute_ade(pred_b, gt_b))
                        fde_total += float(compute_fde(pred_b, gt_b))
                        r_ade_total += float(compute_ade(pred_rel_b, gt_rel_b))
                        r_fde_total += float(compute_fde(pred_rel_b, gt_rel_b))

                        if nsample > 1:
                            # If multi-modal GT is unavailable in batch, fallback to single GT.
                            gt_multi = [gt_b]
                            gt_rel_multi = [gt_rel_b]
                            mmade_total += float(compute_mmade(pred_b, gt_b, gt_multi))
                            mmfde_total += float(compute_mmfde(pred_b, gt_b, gt_multi))
                            r_mmade_total += float(compute_mmade(pred_rel_b, gt_rel_b, gt_rel_multi))
                            r_mmfde_total += float(compute_mmfde(pred_rel_b, gt_rel_b, gt_rel_multi))

                        diversity_total += float(compute_diversity(pred_b))
                        r_diversity_total += float(compute_diversity(pred_rel_b))
                        
                        # Compute smoothness (LDLJ, SPARC, and Mean Acc) using first sample
                        ldlj_total += compute_ldlj(pred_b[0], fps=self.fps)
                        sparc_total += compute_sparc(pred_b[0], fps=self.fps)
                        mean_acc_total += compute_mean_acceleration(pred_b[0])
                        
                        n_sequences += 1
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue

        denom = max(1, n_sequences)
        mmade_avg = (mmade_total / denom) if nsample > 1 else None
        mmfde_avg = (mmfde_total / denom) if nsample > 1 else None
        r_mmade_avg = (r_mmade_total / denom) if nsample > 1 else None
        r_mmfde_avg = (r_mmfde_total / denom) if nsample > 1 else None
        
        m = {
            "AMPJPE": amp_total / denom,
            "FMPJPE": fmp_total / denom,
            "ADE": ade_total / denom,
            "FDE": fde_total / denom,
            "MMADE": mmade_avg,
            "MMFDE": mmfde_avg,
            "Diversity": diversity_total / denom,
            "LDLJ": ldlj_total / denom,
            "SPARC": sparc_total / denom,
            "Mean_Acc": mean_acc_total / denom,
            "R_AMPJPE": r_amp_total / denom,
            "R_FMPJPE": r_fmp_total / denom,
            "R_ADE": r_ade_total / denom,
            "R_FDE": r_fde_total / denom,
            "R_MMADE": r_mmade_avg,
            "R_MMFDE": r_mmfde_avg,
            "R_Diversity": r_diversity_total / denom,
        }

        if samples_list:
            # Flatten all generations into a single sample pool for diversity calculation
            all_s = np.concatenate(samples_list, axis=0)  # (Total_B, N, K, T_out)
            all_s = all_s.reshape(-1, all_s.shape[-2], all_s.shape[-1])  # (Total_Samples, K, T_out)

            
            div_times = min(300, len(all_s) // 2)
            
            m["Div_Spatial"] = get_diversity(all_s.reshape(len(all_s), -1), div_times)
            vel = np.diff(all_s, axis=-1).reshape(len(all_s), -1)
            acc = np.diff(all_s, axis=-1, n=2).reshape(len(all_s), -1)
            m["Div_Vel"] = get_diversity(vel, div_times)
            m["Div_Acc"] = get_diversity(acc, div_times)

        return m
