import pytorch_lightning as pl
import torch
from torch.optim import Adam
from omegaconf import OmegaConf

# Import the model from the copied-over modules
from modules.model import ModelMain
from utils.text_encoder import TextEncoder

class SFTSystem(pl.LightningModule):
    def __init__(self, config, target_dim: int = 72):
        super().__init__()
        # Save config and target_dim so they get recorded in hparams
        self.save_hyperparameters(ignore=['config'])
        self.save_hyperparameters(config)
        self.config = config
        
        # For compatibility with the legacy ModelMain, convert the Hydra DictConfig into a plain Python dict
        cfg_dict = OmegaConf.to_container(config, resolve=True)

        # Switch the underlying generative architecture based on gen_type or model.name
        gen_type = config.get("training", {}).get("gen_type", "ddpm")
        model_name = config.get("model", {}).get("name", "sft_baseline")
        if gen_type == "flow_matching" or model_name == "flow_matching":
            from modules.fm_model import FlowMatchingModel
            self.model = FlowMatchingModel(cfg_dict["model"], device=torch.device('cpu'), target_dim=target_dim)
        else:
            self.model = ModelMain(cfg_dict["model"], device=torch.device('cpu'), target_dim=target_dim)

        
        # Initialize the Text Encoder (does not participate in gradient updates)
        self.text_encoder = TextEncoder(device=torch.device('cpu'))
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, batch, is_train=False, force_dropout=False, force_shuffle=False):
        # If text_encoder has an internal device issue, it can be updated here
        self.text_encoder.device = self.device

        # Extract text from the batch (our Datasets all uniformly return "motion_name")
        texts = batch.get("motion_name", ["unknown"] * batch["pose"].shape[0])
        
        if force_shuffle:
            import random
            texts = list(texts)
            random.shuffle(texts)
            
        # TextEncoder returns a (tok_emb, tok_mask) tuple
        # We pass this tuple directly to ModelMain
        tok_emb, tok_mask = self.text_encoder(texts)
        
        if force_dropout:
            tok_emb = torch.zeros_like(tok_emb)
            
        text_embedding = (tok_emb, tok_mask)
        return self.model(batch, is_train=is_train, text_embedding=text_embedding)

    def training_step(self, batch, batch_idx):
        # Read the CFG dropout probability
        cfg_prob = self.config.training.get("cfg_dropout_prob", 0.05)
        force_drop = (torch.rand(1).item() < cfg_prob)
        
        loss = self(batch, is_train=True, force_dropout=force_drop)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 1. Normal Loss
        loss_main = self(batch, is_train=False)
        self.log('val_loss', loss_main, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # 2. CFG Robustness Check (evaluates how much the model relies on the text)
        with torch.no_grad():
            loss_shuf = self(batch, is_train=False, force_shuffle=True)
            loss_zero = self(batch, is_train=False, force_dropout=True)

        # Higher r_shuf or r_zero means the neural network's loss spikes when it detects the text has been shuffled or is missing
        # This means the model relies heavily on your text conditioning (strong Conditioning Robustness)
        r_shuf = loss_shuf / (loss_main + 1e-9)
        r_zero = loss_zero / (loss_main + 1e-9)
        
        self.log('val_r_shuf', r_shuf, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_r_zero', r_zero, on_step=False, on_epoch=True, sync_dist=True)
        self.log('epoch_1based', float(self.current_epoch + 1), on_step=False, on_epoch=True, sync_dist=True)
        
        return loss_main

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer
