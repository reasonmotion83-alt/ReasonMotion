import pytorch_lightning as pl
import torch
from torch.optim import Adam
from omegaconf import OmegaConf

# 從複製過來的 modules 中引用模型
from modules.model import ModelMain
from utils.text_encoder import TextEncoder

class SFTSystem(pl.LightningModule):
    def __init__(self, config, target_dim: int = 72):
        super().__init__()
        # 儲存 config 與 target_dim 以便 hparams 記錄
        self.save_hyperparameters(ignore=['config'])
        self.save_hyperparameters(config)
        self.config = config
        
        # 為了與舊版 ModelMain 相容，將 Hydra DictConfig 轉為一般 Python dict
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        
        # 根據 gen_type 或 model.name 切換底層生成架構
        gen_type = config.get("training", {}).get("gen_type", "ddpm")
        model_name = config.get("model", {}).get("name", "sft_baseline")
        if gen_type == "flow_matching" or model_name == "flow_matching":
            from modules.fm_model import FlowMatchingModel
            self.model = FlowMatchingModel(cfg_dict["model"], device=torch.device('cpu'), target_dim=target_dim)
        else:
            self.model = ModelMain(cfg_dict["model"], device=torch.device('cpu'), target_dim=target_dim)

        
        # 初始化 Text Encoder (不參與梯度更新)
        self.text_encoder = TextEncoder(device=torch.device('cpu'))
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, batch, is_train=False, force_dropout=False, force_shuffle=False):
        # 如果 text_encoder 內部有 device 問題，可以在這裡更新
        self.text_encoder.device = self.device 
        
        # 從 batch 中提取文字 (我們的 Dataset 都統一回傳 "motion_name")
        texts = batch.get("motion_name", ["unknown"] * batch["pose"].shape[0])
        
        if force_shuffle:
            import random
            texts = list(texts)
            random.shuffle(texts)
            
        # TextEncoder 會回傳 (tok_emb, tok_mask) 的 tuple
        # 我們直接把這個 tuple 傳給 ModelMain
        tok_emb, tok_mask = self.text_encoder(texts)
        
        if force_dropout:
            tok_emb = torch.zeros_like(tok_emb)
            
        text_embedding = (tok_emb, tok_mask)
        return self.model(batch, is_train=is_train, text_embedding=text_embedding)

    def training_step(self, batch, batch_idx):
        # 讀取 CFG dropout 機率
        cfg_prob = self.config.training.get("cfg_dropout_prob", 0.05)
        force_drop = (torch.rand(1).item() < cfg_prob)
        
        loss = self(batch, is_train=True, force_dropout=force_drop)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 1. 正常 Loss
        loss_main = self(batch, is_train=False)
        self.log('val_loss', loss_main, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 2. CFG Robustness Check (評估模型依賴文字的程度)
        with torch.no_grad():
            loss_shuf = self(batch, is_train=False, force_shuffle=True)
            loss_zero = self(batch, is_train=False, force_dropout=True)
            
        # 如果 r_shuf 或 r_zero 越高，代表神經網路發現「文字被亂換了」或者「沒有文字」時 loss 暴增
        # 這意味著模型非常仰賴你的文字條件 (Conditioning Robustness 強)
        r_shuf = loss_shuf / (loss_main + 1e-9)
        r_zero = loss_zero / (loss_main + 1e-9)
        
        self.log('val_r_shuf', r_shuf, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_r_zero', r_zero, on_step=False, on_epoch=True, sync_dist=True)
        self.log('epoch_1based', float(self.current_epoch + 1), on_step=False, on_epoch=True, sync_dist=True)
        
        return loss_main

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer
