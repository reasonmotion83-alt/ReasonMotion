import os
import glob
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", message=".*Found .* module.* in eval mode.*")
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# We use custom step-corrected loggers to shift 0-indexed logging steps to 1-indexed (e.g. 49 -> 50)
from utils.vis import StepCorrectedTensorBoardLogger, StepCorrectedCSVLogger
from torch.utils.data import DataLoader

from train import build_dataset
from systems.grpo_system import GRPOSystem
from utils.vis import VisualizationCallback

@hydra.main(version_base=None, config_path="configs", config_name="train_rl")
def main(cfg: DictConfig):
    if not cfg.get("sft_dir"):
        raise ValueError("An SFT directory must be provided, e.g.: python train_rl.py sft_dir=outputs/YYYY-MM-DD_HH-MM-SS")

    sft_dir = cfg.sft_dir
    sft_config_path = os.path.join(sft_dir, ".hydra", "config.yaml")
    if not os.path.exists(sft_config_path):
        raise FileNotFoundError(f"SFT config file not found: {sft_config_path}")
        
    print(f"📥 Loading SFT base config from {sft_config_path}")
    sft_cfg = OmegaConf.load(sft_config_path)
    
    # Merge RL specific overrides into SFT config
    # RL config overrides SFT config
    merged_cfg = OmegaConf.merge(sft_cfg, cfg)
    
    OmegaConf.set_readonly(merged_cfg, False)
    # Align model.name with training.gen_type
    model_name = merged_cfg.model.get("name", "sft_baseline")
    gen_type = merged_cfg.training.get("gen_type", "ddpm")
    if model_name == "flow_matching" or gen_type == "flow_matching":
        merged_cfg.model.name = "flow_matching"
        merged_cfg.training.gen_type = "flow_matching"
        
    is_quick = merged_cfg.get("quick_test", False)
    if is_quick:
        merged_cfg.training.epochs = 2
        merged_cfg.training.valid_epoch_interval = 1
        if "rl" in merged_cfg:
            merged_cfg.rl.ppo_epochs = 1
            merged_cfg.rl.num_samples_per_prompt = 2
            merged_cfg.rl.update_ref_every_epoch = 1
    
    print("====== RUN CONFIGURATION ======")
    print(OmegaConf.to_yaml(merged_cfg))
    print("===============================")
    
    pl.seed_everything(merged_cfg.system.seed)
    
    data_ratio = 0.01 if is_quick else merged_cfg.dataset.get('data_ratio', 1.0)

    # 1. Build Dataset
    train_ds, target_dim, _ = build_dataset(merged_cfg, split=0, data_ratio=data_ratio)
    valid_ds, _, edges      = build_dataset(merged_cfg, split=1, data_ratio=data_ratio)
    
    # 2. Locate the SFT Checkpoint
    ckpt_dir = os.path.join(sft_dir, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"SFT Checkpoint not found: {ckpt_dir}")

    sft_ckpt_spec = merged_cfg.get("sft_ckpt", None)
    if sft_ckpt_spec:
        sft_ckpt_spec_str = str(sft_ckpt_spec)
        # Support a full path or fuzzy matching
        if os.path.exists(sft_ckpt_spec_str):
            sft_ckpt = sft_ckpt_spec_str
        else:
            # Prefer searching for the exact epoch marker first, to avoid '=' causing Hydra parsing failures
            match_patterns = []
            if sft_ckpt_spec_str.isdigit():
                # Zero-padding support (e.g. entering 30 can match 030)
                zero_padded = f"{int(sft_ckpt_spec_str):03d}"
                match_patterns.extend([
                    f"sft_epoch={zero_padded}-", f"sft_epoch={zero_padded}_",
                    f"moe_epoch={zero_padded}-", f"moe_epoch={zero_padded}_",
                    f"sft-epoch={zero_padded}-", f"sft-epoch={zero_padded}_",
                    f"moe-epoch={zero_padded}-", f"moe-epoch={zero_padded}_",
                ])
                # Matching without zero-padding, as originally
                match_patterns.extend([
                    f"sft_epoch={sft_ckpt_spec_str}-", f"sft_epoch={sft_ckpt_spec_str}_",
                    f"moe_epoch={sft_ckpt_spec_str}-", f"moe_epoch={sft_ckpt_spec_str}_",
                    f"sft-epoch={sft_ckpt_spec_str}-", f"sft-epoch={sft_ckpt_spec_str}_",
                    f"moe-epoch={sft_ckpt_spec_str}-", f"moe-epoch={sft_ckpt_spec_str}_",
                    f"epoch_1based={sft_ckpt_spec_str}-", f"epoch_1based={sft_ckpt_spec_str}_",
                ])
            match_patterns.append(sft_ckpt_spec_str)
            
            matching_ckpts = []
            for pattern in match_patterns:
                matching_ckpts = [c for c in ckpts if pattern in os.path.basename(c)]
                if matching_ckpts:
                    break
                    
            if not matching_ckpts:
                available = [os.path.basename(c) for c in ckpts]
                raise FileNotFoundError(f"No Checkpoint matching '{sft_ckpt_spec}' found in {ckpt_dir}. Available files: {available}")
            sft_ckpt = matching_ckpts[0]
    else:
        # Default to the most recently modified one
        ckpts.sort(key=os.path.getmtime)
        sft_ckpt = ckpts[-1]
    print(f"📥 Found SFT Checkpoint: {sft_ckpt}")
    
    # 3. Build the System (GRPOSystem, DDPOSystem, or PPOSystem)
    algorithm = str(merged_cfg.get("rl", {}).get("algorithm", "grpo")).lower()
    if algorithm == "ddpo":
        from systems.ddpo_system import DDPOSystem
        print("🚀 Initializing DDPOSystem (Strict DDPO RL - Critic-Free with Per-Sample Running Baseline)...")
        system = DDPOSystem(config=merged_cfg, target_dim=target_dim, val_dataset=valid_ds)
    elif algorithm == "ppo":
        from systems.ppo_system import PPOSystem
        print("🚀 Initializing PPOSystem (Actor-Critic PPO RL with Spatial-Temporal Critic)...")
        system = PPOSystem(config=merged_cfg, target_dim=target_dim, val_dataset=valid_ds)
    else:
        from systems.grpo_system import GRPOSystem
        print("🚀 Initializing GRPOSystem (vis-GRPO Critic-Free RL)...")
        system = GRPOSystem(config=merged_cfg, target_dim=target_dim, val_dataset=valid_ds)
    
    # Load SFT weights (only load into the policy model / ref_model)
    import torch
    state_dict = torch.load(sft_ckpt, map_location='cpu')["state_dict"]
    system.load_state_dict(state_dict, strict=False)
    # Manually synchronize policy weights to old_model and ref_model to avoid uninitialized states
    system.old_model.load_state_dict(system.model.state_dict())
    system.ref_model.load_state_dict(system.model.state_dict())
    print(f"✅ SFT Checkpoint loaded and synchronized across model variants!")
    
    train_loader = DataLoader(train_ds, batch_size=merged_cfg.training.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=merged_cfg.training.batch_size_test, shuffle=False, num_workers=4)
    
    # Retrieve output directory directly from HydraConfig
    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Save the merged config to output_dir/.hydra/config.yaml for offline evaluator/visualizer compatibility
    hydra_dir = os.path.join(output_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)
    OmegaConf.save(merged_cfg, os.path.join(hydra_dir, "config.yaml"))
    
    # 4. Set up callbacks and Logger
    new_ckpt_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=new_ckpt_dir,
        filename="rl_step={step:06d}-epoch={epoch_1based:03.0f}-val_loss={val_loss:.4f}",
        save_top_k=merged_cfg.training.get("save_top_k", -1),
        monitor="val_loss",
        mode="min"
    )
    
    vis_callback = VisualizationCallback(
        val_interval=merged_cfg.training.valid_epoch_interval,
        edges=edges
    )
    
    tb_logger = StepCorrectedTensorBoardLogger(save_dir=output_dir, name="tensorboard", version="")
    csv_logger = StepCorrectedCSVLogger(save_dir=output_dir, name="csv_logs", version="")
    logger_list = [tb_logger, csv_logger]
    
    # 5. Launch the Trainer
    val_check_interval = merged_cfg.training.get("val_check_interval", None)
    log_every_n_steps = merged_cfg.training.get("log_every_n_steps", 50)
    
    check_val_every_n_epoch = merged_cfg.training.valid_epoch_interval
    if val_check_interval is not None:
        check_val_every_n_epoch = None

    trainer = pl.Trainer(
        accelerator=merged_cfg.system.accelerator,
        devices=merged_cfg.system.devices,
        strategy=merged_cfg.system.strategy,
        precision=merged_cfg.system.precision,
        max_epochs=merged_cfg.training.epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=merged_cfg.training.get("limit_val_batches", None),
        callbacks=[checkpoint_callback, vis_callback],
        logger=logger_list,
    )
    
    trainer.fit(system, train_loader, valid_loader)

if __name__ == "__main__":
    main()
