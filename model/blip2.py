"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os

import torch
import torch.nn as nn, torch.nn.functional as F

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer, BertModel
try:    
    from flash_attn.models.bert import BertForPreTraining
except TypeError:
    BertForPreTraining = None
from .cell_encoder import CellEncoder

class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim=256):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, bert_model_name):
        if bert_model_name == "scibert":
            bert_name = 'allenai/scibert_scivocab_uncased'
            print("bert load scibert as tokenizer")
        elif bert_model_name == 'pubmedbert':
            bert_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
            print("bert load pubmedbert as tokenizer")
        else:
            bert_name = 'bert_pretrained/'
        tokenizer = BertTokenizer.from_pretrained(bert_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, model_name, num_query_token, graph_width, cross_attention_freq=2, use_flash_attn = False):
        assert model_name == 'scibert' or 'pubmedbert' #pubmedbert

        if model_name == 'scibert':
            bert_name = 'allenai/scibert_scivocab_uncased'
            print("bert load scibert")
        elif model_name == 'pubmedbert':
            bert_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
            print("bert load pubmedbert")
        else:
            bert_name = 'bert_pretrained/'
    
        
        encoder_config = BertConfig.from_pretrained(bert_name)
        encoder_config.encoder_width = graph_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained(
            bert_name, config=encoder_config
        )
        if use_flash_attn:
            encoder_config.use_flash_attn = True
            Qformer.bert = BertForPreTraining.from_pretrained(encoder_config, add_pooling_layer=False).bert
        else:
            pass
            
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    

    @classmethod
    def init_cell_encoder(cls, vocab_path, model_path, model_config_path):
        cell_encoder = CellEncoder(vocab_path, model_path, model_config_path)
        ln_gene = LayerNorm(cell_encoder.d_model)
        return cell_encoder, ln_gene

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
