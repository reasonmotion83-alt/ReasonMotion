import os
import warnings

# ==========================================
# 📊 View training curves (TensorBoard) command:
# Open a new terminal, and from the project root run:
# tensorboard --logdir outputs --bind_all --port 6006
# (this will automatically load all training logs under the outputs folder)
# ==========================================

# Disable huggingface tokenizers' annoying fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable TensorFlow's annoying C++ logs and oneDNN warnings (pulled in by TensorBoardLogger)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ignore PyTorch Lightning's warning about TextEncoder being in eval mode
warnings.filterwarnings("ignore", message=".*Found .* module.* in eval mode.*")
# Ignore the Protobuf version mismatch warning
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Import the legacy Dataset (already copied from motion_data to the data folder)
from data.h36m_unified import H36MUnified
from data.finefs import FineFS

# Import the Lightning System we just created
from systems.sft_system import SFTSystem

def build_dataset(cfg, split, data_ratio=1.0):
    """Build the corresponding dataset based on the Dataset settings"""
    ds_name = cfg.dataset.name.lower()
    common_kw = dict(
        input_n    = cfg.dataset.input_n,
        output_n   = cfg.dataset.output_n,
        skip_rate  = cfg.dataset.skip_rate,
        split      = split,
    )
    if split == 1:
        # Validation
        common_kw['skip_rate'] = cfg.dataset.get('val_skip_rate', 25)
        
    is_quick = cfg.get("quick_test", False)
    ratio = 0.01 if is_quick else cfg.dataset.get('data_ratio', 1.0)
    
    if ds_name == 'finefs':
        from data.finefs import EDGES
        return FineFS(
            data_dir = cfg.dataset.data_dir,
            mode     = cfg.dataset.get('mode', 'rotation'),
            random_face = cfg.dataset.get('random_face', False) if split == 0 else False,
            filter_single_rotation = cfg.dataset.get('filter_single_rotation', False),
            disable_sliding = cfg.dataset.get('disable_sliding', False),
            max_len  = cfg.dataset.get('max_len', 90),
            data_ratio = ratio,
            **common_kw
        ), 24 * 3, EDGES
    elif ds_name == 'h36m':
        from data.h36m import EDGES_H36M
        joints = cfg.dataset.get('joints', 22)
        return H36MUnified(
            data_dir=cfg.dataset.data_dir,
            joints=joints,
            downsample=cfg.dataset.get('downsample', 2),
            no_overlap=cfg.dataset.get('no_overlap', False),
            protocol=cfg.dataset.get('protocol', 'allen'),
            miss_type=cfg.dataset.get('miss_type', 'no_miss'),
            miss_rate=cfg.dataset.get('miss_rate', 0.2),
            all_data=cfg.dataset.get('all_data', True),
            data_ratio=ratio,
            pad_short_sequences=cfg.dataset.get('pad_short_sequences', False),
            **common_kw,
        ), joints * 3, EDGES_H36M
    elif ds_name == 'boxing':
        from data.boxing import BoxingDataset
        from data.skeleton import EDGES_BODY25
        return BoxingDataset(
            data_dir=cfg.dataset.data_dir,
            random_face=cfg.dataset.get('random_face', False) if split == 0 else False,
            data_ratio=ratio,
            neutral_ratio=cfg.dataset.get('neutral_ratio', 1.5),
            **common_kw,
        ), 25 * 3, EDGES_BODY25
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")



@hydra.main(version_base=None, config_path="configs", config_name="train_sft")
def main(cfg: DictConfig):
    OmegaConf.set_readonly(cfg, False)
    
    # Align model.name with training.gen_type
    model_name = cfg.model.get("name", "sft_baseline")
    gen_type = cfg.training.get("gen_type", "ddpm")
    if model_name == "flow_matching" or gen_type == "flow_matching":
        cfg.model.name = "flow_matching"
        cfg.training.gen_type = "flow_matching"

    is_quick = cfg.get("quick_test", False)
    if is_quick:
        cfg.training.epochs = 2
        cfg.training.valid_epoch_interval = 1

    # Print out the final resolved Config
    print("====== RUN CONFIGURATION ======")
    print(OmegaConf.to_yaml(cfg))
    print("===============================")
    
    pl.seed_everything(cfg.system.seed)
    
    # 1. Build Dataset & DataLoader
    train_ds, target_dim, _ = build_dataset(cfg, split=0)
    valid_ds, _, edges      = build_dataset(cfg, split=1)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(valid_ds)}")
    print(f"Target Dimension: {target_dim}")
    
    # num_workers can be adjusted dynamically based on hardware; set to 4 for now
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.training.batch_size_test, shuffle=False, num_workers=4)
    
    # 2. Build the PyTorch Lightning system
    system = SFTSystem(config=cfg, target_dim=target_dim)
    
    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    
    # 3. Set up callbacks and Logger
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    is_moe = cfg.model.get("moe", {}).get("enabled", False)
    prefix = "moe_epoch" if is_moe else "sft_epoch"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{prefix}={{epoch_1based:03.0f}}-val_loss={{val_loss:.4f}}",
        save_top_k=cfg.training.get("save_top_k", -1),
        monitor="val_loss",
        mode="min"
    )
    
    from utils.vis import VisualizationCallback
    vis_callback = VisualizationCallback(
        val_interval=cfg.training.valid_epoch_interval,
        edges=edges
    )
    
    from utils.vis import StepCorrectedTensorBoardLogger, StepCorrectedCSVLogger
    
    tb_logger = StepCorrectedTensorBoardLogger(save_dir=output_dir, name="tensorboard", version="")
    csv_logger = StepCorrectedCSVLogger(save_dir=output_dir, name="csv_logs", version="")
    
    logger_list = [tb_logger, csv_logger]
    # 4. Launch the Trainer
    trainer = pl.Trainer(
        accelerator=cfg.system.accelerator,
        devices=cfg.system.devices,
        strategy=cfg.system.strategy,
        precision=cfg.system.precision,
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=cfg.training.valid_epoch_interval,
        callbacks=[checkpoint_callback, vis_callback],
        logger=logger_list,
        gradient_clip_val=cfg.training.grad_clip,
    )
    
    # Start training!
    trainer.fit(system, train_loader, valid_loader)

if __name__ == "__main__":
    main()
