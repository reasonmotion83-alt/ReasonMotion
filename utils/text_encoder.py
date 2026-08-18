# utils/text_encoder.py
from typing import List, Tuple
import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """
    Converts list[str] → (tok_emb, tok_mask)
        tok_emb  : (B,T,384)
        tok_mask : (B,T)  1 = real token, 0 = pad
    """
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu",
                 max_len: int = 12):
        super().__init__()
        self.sbert = SentenceTransformer(model_name, device=device)
        self.device = torch.device(device)
        self.max_len = max_len

        # Get the BERT backbone
        self.bert = self.sbert[0].auto_model        # transformers.PreTrainedModel
        self.tokenizer = self.sbert.tokenizer

    @torch.no_grad()
    def forward(self, sentences: List[str]) -> Tuple[Tensor, Tensor]:
        device = next(self.bert.parameters()).device
        tok = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        ).to(device)                        # input_ids, attention_mask

        # BERT forward → hidden state (B,T,384)
        out = self.bert(
            input_ids      = tok["input_ids"],
            attention_mask = tok["attention_mask"],
            token_type_ids = tok.get("token_type_ids", None),
        )
        tok_emb  = out.last_hidden_state         # (B,T,384)
        tok_mask = tok["attention_mask"]         # (B,T)

        return tok_emb, tok_mask
