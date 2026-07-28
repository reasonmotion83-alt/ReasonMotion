import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
ReasonMotion 離線量化評估腳本 (Offline Evaluator)
此腳本載入指定的 Checkpoint 世代，並在測試集 (split=2) 上進行定量指標評估。

使用範例：
    python scripts/evaluate.py --exp_dir outputs/sft_finefs_2026-07-13_12-24 --epoch 140 --nsample 5 --batch_size 32
"""
import glob
import re
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from systems.sft_system import SFTSystem
from systems.grpo_system import GRPOSystem
from utils.metrics import MetricsEvaluator
from train import build_dataset


def get_target_checkpoint(args, ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        raise ValueError(f"找不到任何 .ckpt 於 {ckpt_dir}")
        
    def get_epoch_or_step(path):
        name = os.path.basename(path)
        m_sft = re.search(r"sft_epoch=(\d+)", name)
        if m_sft: return int(m_sft.group(1))
        m_moe = re.search(r"moe_epoch=(\d+)", name)
        if m_moe: return int(m_moe.group(1))
        m_rl_ep = re.search(r"rl_epoch=(\d+)", name)
        if m_rl_ep: return int(m_rl_ep.group(1))
        m_rl_step = re.search(r"rl_step=(\d+)", name)
        if m_rl_step: return int(m_rl_step.group(1))
        m1 = re.search(r"epoch_1based=(\d+)", name)
        if m1: return int(m1.group(1))
        m2 = re.search(r"epoch=(\d+)", name)
        if m2: return int(m2.group(1)) + 1
        m3 = re.search(r"step=(\d+)", name)
        if m3: return int(m3.group(1))
        return -1

    # 1. 優先匹配 --step 參數 (針對 RL 步數)
    target_step = getattr(args, "step", -1)
    if target_step != -1:
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            m_step = re.search(r"rl_step=(\d+)", name) or re.search(r"step=(\d+)", name)
            if m_step and int(m_step.group(1)) == target_step:
                return ckpt
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            if f"step={target_step}" in name or f"rl_step={target_step}" in name:
                return ckpt
                
    # 2. 次要匹配 --epoch 參數
    target_epoch = getattr(args, "epoch", -1)
    if target_epoch != -1:
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            m_ep = (re.search(r"sft_epoch=(\d+)", name) or 
                    re.search(r"moe_epoch=(\d+)", name) or 
                    re.search(r"epoch_1based=(\d+)", name) or 
                    re.search(r"epoch=(\d+)", name))
            if m_ep:
                val = int(m_ep.group(1))
                if "epoch=" in name and not any(k in name for k in ["sft_epoch", "moe_epoch", "epoch_1based", "rl-epoch"]):
                    val += 1
                if val == target_epoch:
                    return ckpt
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            if (f"epoch={target_epoch}" in name or 
                f"epoch_1based={target_epoch}" in name or 
                f"sft_epoch={target_epoch}" in name or 
                f"moe_epoch={target_epoch}" in name):
                return ckpt
        raise ValueError(f"在 {ckpt_dir} 中找不到對應 Epoch/Step {target_epoch} 的 Checkpoint！可用檔案：{[os.path.basename(c) for c in ckpts]}")
    
    ckpts_sorted = sorted(ckpts, key=os.path.getmtime)
    return ckpts_sorted[-1]


def build_model_from_checkpoint(config, ckpt_path, device):
    print(f"📥 載入權重: {ckpt_path}")
    ds_name = config.get("dataset", {}).get("name", "").lower()
    if ds_name == "h36m":
        target_dim = config.get("dataset", {}).get("joints", 17) * 3
    elif ds_name == "boxing":
        target_dim = 25 * 3
    else:
        target_dim = 24 * 3

    
    import torch as _torch
    is_rl = "kl_coef" in config
    if is_rl:
        system = GRPOSystem(config=config, target_dim=target_dim)
    else:
        system = SFTSystem(config=config, target_dim=target_dim)
    
    ckpt = _torch.load(ckpt_path, map_location="cpu")
    system.load_state_dict(ckpt["state_dict"], strict=False)
    
    system.to(device)
    system.eval()
    return system


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def close(self):
        if self.log and not self.log.closed:
            self.log.close()


def main():
    # 預先解析 --config / -c 參數
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "-c", type=str, default="configs/eval.yaml", help="Path to evaluation config YAML file")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # 定義超參數預設值
    cfg_defaults = {
        "exp_dir": None,
        "epoch": -1,
        "step": -1,
        "ckpt": "",
        "batch_size": 32,
        "nsample": 1,
        "device": "cuda",
        "quick": False,
        "sample_ratio": 1.0,
        "save_dir": "",
        "no_save": False,
    }

    config_yaml_path = pre_args.config
    if config_yaml_path and os.path.exists(config_yaml_path):
        try:
            loaded_cfg = OmegaConf.load(config_yaml_path)
            for k, v in loaded_cfg.items():
                if k in cfg_defaults and v is not None:
                    cfg_defaults[k] = v
        except Exception as e:
            print(f"⚠️ 讀取設定檔 ({config_yaml_path}) 失敗: {e}")

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint quantitatively")
    parser.add_argument("--config", "-c", type=str, default=config_yaml_path, help="Path to evaluation config YAML file")
    parser.add_argument("--exp_dir", type=str, default=cfg_defaults["exp_dir"], help="Path to experiment output directory")
    parser.add_argument("--epoch", type=int, default=cfg_defaults["epoch"], help="Specific epoch checkpoint to load (1-based)")
    parser.add_argument("--step", type=int, default=cfg_defaults["step"], help="Specific step checkpoint to load (RL-only)")
    parser.add_argument("--ckpt", type=str, default=cfg_defaults["ckpt"], help="Specific path to a .ckpt file (overrides --epoch/--step)")
    parser.add_argument("--batch_size", type=int, default=cfg_defaults["batch_size"], help="Batch size for evaluation")
    parser.add_argument("--nsample", type=int, default=cfg_defaults["nsample"], help="Number of samples to generate per prompt (best-of-N if > 1)")
    parser.add_argument("--device", type=str, default=cfg_defaults["device"], help="Device to run evaluation (cuda/cpu)")
    parser.add_argument("--quick", action="store_true", default=cfg_defaults["quick"], help="Enable quick test mode (evaluates on only 1%% of the dataset)")
    parser.add_argument("--sample_ratio", "--ratio", type=float, default=cfg_defaults["sample_ratio"], help="Evaluate on a subset ratio of test dataset (0.0 to 1.0, e.g. 0.1 for 10%%)")
    parser.add_argument("--save_dir", type=str, default=cfg_defaults["save_dir"], help="Custom output directory for evaluation results (default: <exp_dir>/eval)")
    parser.add_argument("--no_save", action="store_true", default=cfg_defaults["no_save"], help="Disable saving evaluation results to file")
    
    args = parser.parse_args(remaining_argv)


    if not args.exp_dir:
        parser.error("必須指定 --exp_dir 或是於 configs/eval.yaml 中設定 exp_dir。")

    args.exp_dir = os.path.abspath(args.exp_dir)

    config_path = os.path.join(args.exp_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到 Hydra 設定檔於: {config_path}")
    config = OmegaConf.load(config_path)
    
    OmegaConf.set_readonly(config, False)
    model_name = config.model.get("name", "sft_baseline")
    gen_type = config.training.get("gen_type", "ddpm")
    if model_name == "flow_matching" or gen_type == "flow_matching":
        config.model.name = "flow_matching"
        config.training.gen_type = "flow_matching"

    device_str = str(args.device).strip()
    if device_str.isdigit():
        device_str = f"cuda:{device_str}"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")



    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
        ckpt_path = get_target_checkpoint(args, ckpt_dir)

    tee_logger = None
    eval_dir = None
    filename_stem = None
    log_path = None
    json_path = None

    if not args.no_save:
        eval_dir = args.save_dir if args.save_dir else os.path.join(args.exp_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        ckpt_name = os.path.basename(ckpt_path)
        if args.step != -1:
            filename_stem = f"eval_step{args.step}"
        elif args.epoch != -1:
            filename_stem = f"eval_epoch{args.epoch}"
        else:
            m_step = re.search(r"rl_step=(\d+)", ckpt_name) or re.search(r"step=(\d+)", ckpt_name)
            m_ep = (re.search(r"rl_epoch=(\d+)", ckpt_name) or 
                    re.search(r"sft_epoch=(\d+)", ckpt_name) or 
                    re.search(r"moe_epoch=(\d+)", ckpt_name) or 
                    re.search(r"epoch_1based=(\d+)", ckpt_name) or 
                    re.search(r"epoch=(\d+)", ckpt_name))
            if m_step:
                filename_stem = f"eval_step{int(m_step.group(1))}"
            elif m_ep:
                filename_stem = f"eval_epoch{int(m_ep.group(1))}"
            else:
                filename_stem = f"eval_{os.path.splitext(ckpt_name)[0]}"

        if args.quick:
            filename_stem += "_quick"

        log_path = os.path.join(eval_dir, f"{filename_stem}.log")
        json_path = os.path.join(eval_dir, f"{filename_stem}.json")
        tee_logger = TeeLogger(log_path)
        sys.stdout = tee_logger

    try:
        system = build_model_from_checkpoint(config, ckpt_path, device)

        print("📂 載入測試資料集 (Split 2)...")
        if args.quick:
            if "training" not in config:
                config.training = {}
            config.training.quick_test = True
            config.quick_test = True
            if "dataset" in config:
                config.dataset.max_len = 15
        else:
            if "training" in config and "quick_test" in config.training:
                config.training.quick_test = False
            config.quick_test = False
        test_dataset, _, _ = build_dataset(config, split=2)
        
        if 0.0 < args.sample_ratio < 1.0:
            num_samples = max(1, int(len(test_dataset) * args.sample_ratio))
            print(f"✂️ 套用資料比例 --sample_ratio {args.sample_ratio:.2f}：從 {len(test_dataset)} 個樣本中選取 {num_samples} 個樣本評估。")
            indices = list(range(num_samples))
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )


        evaluator = MetricsEvaluator(config=config, device=device)
        if hasattr(system, "text_encoder"):
            evaluator.text_encoder = system.text_encoder

        print(f"🚀 開始評估量化指標 (n_sample={args.nsample}, 樣本總數={len(test_dataset)})...")
        metrics = evaluator.evaluate(
            model=system.model,
            dataloader=dataloader,
            nsample=args.nsample
        )

        print("\n" + "=" * 60)
        print(f"📊 Quantitative Evaluation Results for: {os.path.basename(ckpt_path)}")
        print("=" * 60)
        print(f"{'Metric Name':<25} | {'Value':<20}")
        print("-" * 60)
        for key, value in metrics.items():
            if value is None:
                continue
            print(f"{key:<25} | {value:.6f}")
        print("=" * 60 + "\n")

        if eval_dir and filename_stem and json_path:
            import json
            clean_metrics = {}
            for k, v in metrics.items():
                if v is not None:
                    clean_metrics[k] = float(v) if isinstance(v, (np.floating, np.integer, float, int)) else str(v)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(clean_metrics, f, indent=4, ensure_ascii=False)
            print(f"💾 指標 JSON 已自動儲存至: {json_path}")
            print(f"💾 評估 Log 已自動儲存至: {log_path}\n")

    except KeyboardInterrupt:
        print("\n❌ 評估被使用者中斷。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 評估失敗: {e}")
    finally:
        if tee_logger is not None:
            sys.stdout = tee_logger.terminal
            tee_logger.close()


if __name__ == "__main__":
    main()

