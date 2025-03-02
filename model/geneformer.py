from datasets import load_from_disk
import torch.nn as nn, torch.nn.functional as F
import torch
from transformers import BertModel
from torch.utils.data import DataLoader

class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output

def cell_encode(model, cell_tokens):
    cell_input_ids = cell_tokens['input_ids']
    cell_atts = cell_tokens['attention_mask']
    cell = model(cell_input_ids.to("cuda"), cell_atts.to("cuda"))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler

def load_model(model_path, proj_path):
    model = BertModel.from_pretrained(model_path)
    model.pooler = Pooler(model.config, pretrained_proj=proj_path, proj_dim=256)
    return model

if __name__=='__main__':
    model = BertModel.from_pretrained('langcell_ckpt/cell_bert')
    model.pooler = Pooler(model.config, pretrained_proj='langcell_ckpt/cell_proj.bin', proj_dim=256)
