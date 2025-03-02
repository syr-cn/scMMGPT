"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast as autocast
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from pprint import pprint
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import LlamaTokenizer, AutoModelForCausalLM, LlamaForCausalLM as LlamaForCausalLMOriginal
# from model.modeling_llama_new import LlamaForCausalLM
from model.modeling_llama import LlamaForCausalLM

from scgpt.model import ExprDecoder

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input

def process_row(
    row,
    normalize_total,
    log1p: bool = False,
    flag: bool = False
) -> torch.Tensor:
    """
    处理单行张量，支持跳过首元素的标准化和log转换
    
    Args:
        row: 输入的一维张量
        normalize_total: 标准化目标总和，None表示不标准化
        log1p: 是否进行log1p转换
        flag: 是否跳过第一个元素处理
    
    Returns:
        处理后的新张量（浮点类型）
    """
    # 创建拷贝并转换为浮点类型
    processed = row.clone().to(torch.float32)
    
    # 分割张量
    if flag:
        head = processed[:, :1]  # 保持第一个元素不变
        tail = processed[:, 1:]
    else:
        head = torch.empty(0, dtype=torch.float32, device=processed.device)
        tail = processed
    
    # 标准化处理
    if normalize_total is not None and tail.numel() > 0:
        tail_sum = tail.sum()
        if tail_sum > 0:  # 仅当有有效数据时标准化
            tail *= (normalize_total / tail_sum)
    
    # log1p转换
    if log1p and tail.numel() > 0:
        # 确保没有负值（标准化后理论上不会有负值）
        tail[tail < 0] = 0
        tail = torch.log1p(tail)
    
    # 合并结果
    if flag:
        return torch.cat([head, tail], dim=1)
    return tail

import torch

def modify_values_by_sorted_rank_tensor(
    batch: torch.Tensor,
    slope: float,
    intercept: float
) -> torch.Tensor:
    """
    
    Args:
        batch: 形状为(batch_size, n_genes)的基因表达值张量
        slope: 线性模型的斜率参数
        intercept: 线性模型的截距参数
    
    Returns:
        重构后的基因表达值张量，保持原始维度
    """
    device = batch.device
    dtype = batch.dtype
    
    # 降序排序获取索引
    sorted_values, sorted_indices = torch.sort(batch, dim=1, descending=True)
    
    # 创建排名矩阵
    n_samples, n_genes = batch.shape
    ranks = torch.zeros_like(batch, dtype=torch.float32, device=device)
    
    # 生成log排名值（使用向量化操作）
    log_ranks = torch.log10(1 + torch.arange(1, n_genes+1, device=device, dtype=torch.float32))
    
    # 使用scatter填充排名值
    rows = torch.arange(n_samples, device=device)[:, None].expand(-1, n_genes)
    ranks.scatter_(dim=1, index=sorted_indices, src=log_ranks.unsqueeze(0).expand(n_samples, -1))
    
    # 应用线性模型
    modified_values = intercept + slope * ranks
    # modified_values = torch.expm1(modified_values)

    # 保持原始数据类型
    return modified_values.to(dtype=dtype)

class MLP_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

BOC_TOKEN = '<cell>'
CELL_TOKEN = '<cell_{:05d}>'
EOC_TOKEN = '</cell>'

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Llama(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        tune_gene_encoder=False,
        num_query_token=32,
        cross_attention_freq=2,
        lora_tuning=False,
        peft_dir='',
        llm_model="decapoda-research/llama-7b-hf",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.valrank = args.valrank
        self.cell_encoder, self.ln_gene = self.init_cell_encoder(
            args.vocab_path,
            args.model_path,
            args.model_config_path
        )
        self.tune_gene_encoder = tune_gene_encoder
        if not tune_gene_encoder:
            for name, param in self.cell_encoder.named_parameters():
                param.requires_grad = False
            self.cell_encoder = self.cell_encoder.eval()
            self.cell_encoder.train = disabled_train
            logging.info("freeze graph encoder")

        print(f"{self.cell_encoder.model=}")
            
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.cell_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.tune_Qformer = args.tune_Qformer
        if not self.tune_Qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            logging.info("freeze Q-former")
        ## initialize opt model
        # self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        # print(self.llm_tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
        # self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        # self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        # self.llm_tokenizer.add_special_tokens({'boc_token': BOC_TOKEN})
        # for item in range(num_query_token):
        #     cell_token = CELL_TOKEN.format(int(item))
        #     self.llm_tokenizer.add_special_tokens({f'cell_token_{item}': cell_token})
        # self.llm_tokenizer.add_special_tokens({'eoc_token': EOC_TOKEN})

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})

        # Use 'additional_special_tokens' for custom tokens
        additional_tokens = [BOC_TOKEN]  # Initialize the list with BOC_TOKEN
        for item in range(num_query_token):
            cell_token = CELL_TOKEN.format(int(item))
            additional_tokens.append(cell_token)
        additional_tokens.append(EOC_TOKEN)

        self.use_sentence = args.use_sentence
        # 如果use_sentence为True，添加gene特殊token
        if self.use_sentence:
            gene_tokens = [f"<gene_{i:05d}>" for i in range(70001)]  # 添加00000到70000的token
            additional_tokens.extend(gene_tokens)

        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})

        print("Special tokens len:", len(self.llm_tokenizer.special_tokens_map))
        print("Additional special tokens len:", len(self.llm_tokenizer.additional_special_tokens))

        # 创建gene_id到token_id的映射表
        if self.use_sentence:
            self.gene_id_to_token_id = {}
            for i in range(70001):
                token = f"<gene_{i:05d}>"
                token_id = self.llm_tokenizer.convert_tokens_to_ids(token)
                self.gene_id_to_token_id[i] = token_id
            max_gene_id = 70000
            gene_id_tensor = torch.zeros(max_gene_id + 1, dtype=torch.long)
            for gene_id, token_id in self.gene_id_to_token_id.items():
                gene_id_tensor[gene_id] = token_id
            self.register_buffer('gene_id_to_token_id_tensor', gene_id_tensor)

        precision = {
            '32': torch.float32,
            '16': torch.float16,
            'bf16': torch.bfloat16,
        }[args.precision]
        if 'tinyllama' in llm_model.lower():
            self.llm_model = LlamaForCausalLMOriginal.from_pretrained(llm_model, torch_dtype=precision)
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=precision)
        # self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        # self.llm_model = LlamaForCausalLM.from_pretrained(llm_model)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        self.lora_tuning = lora_tuning
        if lora_tuning:
            if peft_dir:
                self.llm_model = PeftModel.from_pretrained(self.llm_model, peft_dir, is_trainable=True)
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.llm_tokenizer("</s>", add_special_tokens=False).input_ids[0]
        self.pad_token_id = self.llm_tokenizer.pad_token_id

        # self.llm_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        # )
        self.llm_proj = MLP_Model(self.Qformer.config.hidden_size, 256, self.llm_model.config.hidden_size)
        # self.decoder_proj = MLP_Model(self.llm_model.config.hidden_size, 256, self.llm_model.config.hidden_size)
        # self.decoder_proj = MLP_Model(self.llm_model.config.hidden_size, 256, self.llm_model.config.hidden_size)
        # self.train_decoder = args.train_decoder
        self.train_decoder = True # TODO: add args
        self.expr_decoder = GeneExpressionModel(70000, 512, self.llm_model.config.hidden_size, 512)
        print(f"\n init expr decoder, {self.llm_model.config.hidden_size=}\n")
        # self.expr_decoder = self.cell_encoder.model.decoder

    def forward(self, batch):
        text_tokens, gene_inputs = batch
        if self.valrank:
            slope = -0.6756886579917628
            intercept = 2.237853249084892
            gene_inputs["values"] = modify_values_by_sorted_rank_tensor(gene_inputs["values"],
                                                                slope=slope,
                                                                intercept=intercept)
        
        inputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
        batch_size, seq_len, lm_emb_dim = inputs_embeds.shape

        cell_outputs = self.cell_encoder(gene_inputs)
        if not self.tune_gene_encoder:
            gene_features = cell_outputs.detach()

        gene_features = self.ln_gene(gene_features)
        gene_padding_mask = gene_inputs['padding_mask'].to(gene_features.device)
        gene_attention_mask = torch.logical_not(gene_padding_mask).long()
        
        device = gene_features.device
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=gene_features,
            encoder_attention_mask=gene_attention_mask,
            use_cache=True,
            return_dict=True,
        )

        if not self.tune_Qformer:
            query_output.last_hidden_state = query_output.last_hidden_state.detach()

        cell_inputs_llm = self.llm_proj(query_output.last_hidden_state)

        has_cell_input = text_tokens.cmp_mask.sum().item() > 0 # text_tokens.cmp_mask: [batch_size, seq_len]
        has_cell_output = text_tokens.gen_mask.sum().item() > 0 # text_tokens.gen_mask: [batch_size, seq_len]

        if has_cell_input:
            if self.use_sentence:
                values = gene_inputs["values"]
                gene_ids = gene_inputs["gene_ids"]
                # 使用topk高效获取前k大值的索引
                num_query_token = cell_inputs_llm.shape[1]  # 动态获取当前batch的num_query_token
                topk_values, topk_indices = torch.topk(values, k=num_query_token, dim=1, sorted=True)
                topk_gene_ids = torch.gather(gene_ids, dim=1, index=topk_indices)
                
                # 转换为token ids并获取嵌入
                token_ids = self.gene_id_to_token_id_tensor[topk_gene_ids]
                gene_token_embeds = self.llm_model.get_input_embeddings()(token_ids)

                have_cmp_mask = torch.nonzero(text_tokens.cmp_mask, as_tuple=True)[0].unique()
                inputs_embeds[text_tokens.cmp_mask] = gene_token_embeds[have_cmp_mask].reshape(-1, lm_emb_dim)
            else:
                have_cmp_mask = torch.nonzero(text_tokens.cmp_mask, as_tuple=True)[0].unique()
                inputs_embeds[text_tokens.cmp_mask] = cell_inputs_llm[have_cmp_mask].reshape(-1, lm_emb_dim)
        else:
            inputs_embeds[:1, :query_output.last_hidden_state.shape[1], :] += 0.0 * cell_inputs_llm[:1, :, :]

        # 后续代码保持不变...
        lm_outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=text_tokens.labels,
        )
        lm_loss = lm_outputs.loss

        lm_last_hidden_state = lm_outputs.hidden_states[-1]
        num_query_tokens = cell_inputs_llm.shape[1]
        
        if has_cell_output:
            have_gen_mask = torch.nonzero(text_tokens.gen_mask, as_tuple=True)[0].unique()
            origin_embeds = cell_inputs_llm[have_gen_mask]
            output_cell_embeds = lm_last_hidden_state[text_tokens.gen_mask]
            num_active_samples = have_gen_mask.shape[0]
            rec_cell_embeds = output_cell_embeds.view(num_active_samples, -1, lm_emb_dim)
            rec_loss = F.mse_loss(rec_cell_embeds, origin_embeds.detach())
        else:
            output_cell_embeds = torch.randn(1, num_query_tokens, lm_emb_dim).to(device) + 0.0 * lm_last_hidden_state[0, :num_query_tokens, :]
            rec_loss = 0.0 * output_cell_embeds.sum()

        total_loss = lm_loss + rec_loss

        if self.train_decoder:
            cell_inputs_llm = cell_inputs_llm.detach()
            decoder_output = self.expr_decoder(cell_inputs_llm[:,:1,:].squeeze(1), gene_inputs['gene_ids'])
            gene_values = gene_inputs['values'].to(torch.float32)
            values1p = process_row(gene_values, 1e2, True, True)
            rec_expr_loss = F.mse_loss(decoder_output, values1p.detach())

        total_loss = total_loss + rec_expr_loss.sum()
        return {'loss': total_loss, 'lm_loss': lm_loss, 'rec_loss': rec_loss, 'expr_loss': rec_expr_loss}


    def forward_old(self, batch):

        text_tokens, gene_inputs = batch
        if self.valrank:
            slope = -0.6756886579917628
            intercept = 2.237853249084892
            gene_inputs["values"] = modify_values_by_sorted_rank_tensor(gene_inputs["values"],
                                                                slope=slope,
                                                                intercept=intercept)
        
        inputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids) # [batch_size, seq_len, lm_emb_dim]
        batch_size, seq_len, lm_emb_dim = inputs_embeds.shape
        # print(f"{batch_size=}, {seq_len=}, {lm_emb_dim=}")

        cell_outputs = self.cell_encoder(gene_inputs) # [batch_size, seq_len, d_model]
        if not self.tune_gene_encoder:
            gene_features = cell_outputs.detach()

        gene_features = self.ln_gene(gene_features)
        gene_padding_mask = gene_inputs['padding_mask'].to(gene_features.device) # [batch_size, seq_len], 0 if not padding, 1 if padding
        gene_attention_mask = torch.logical_not(gene_padding_mask).long() # [batch_size, seq_len], 1 if not padding, 0 if padding
        
        device = gene_features.device
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=gene_features,
            encoder_attention_mask=gene_attention_mask,
            use_cache=True,
            return_dict=True,
        )

        if not self.tune_Qformer:
            query_output.last_hidden_state = query_output.last_hidden_state.detach()

        cell_inputs_llm = self.llm_proj(query_output.last_hidden_state) # shape = [B, num_q, D]

        # print(f"{cell_inputs_llm.shape=}")

        """
        atts_llm = torch.ones(cell_inputs_llm.size()[:-1], dtype=torch.long).to(device)
        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100)
        empty_targets = (torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100))
        targets = torch.cat([empty_targets, targets], dim=1)
        attention_mask = torch.cat([atts_llm, text_tokens.attention_mask], dim=1)
        """

        has_cell_input = text_tokens.cmp_mask.sum().item() > 0 # text_tokens.cmp_mask: [batch_size, seq_len]
        has_cell_output = text_tokens.gen_mask.sum().item() > 0 # text_tokens.gen_mask: [batch_size, seq_len]

        if has_cell_input: ##
            # inputs_embeds[text_tokens.cmp_mask] = cell_inputs_llm.reshape(-1, lm_emb_dim)
            have_cmp_mask = torch.nonzero(text_tokens.cmp_mask, as_tuple=True)[0].unique()
            
            inputs_embeds[text_tokens.cmp_mask] = cell_inputs_llm[have_cmp_mask].reshape(-1, lm_emb_dim)
        else:
            inputs_embeds[:1, :query_output.last_hidden_state.shape[1], :] += 0.0 * cell_inputs_llm[:1, :, :]
        
        # print(f"{text_tokens.input_ids[0]=}", flush=True)
        # print(f"{text_tokens.attention_mask[0]=}", flush=True)
        # print(f"{text_tokens.labels[0]=}", flush=True)
        # # print(f"{text_tokens.input_ids=}", flush=True)
        # assert(False)

        lm_outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=text_tokens.labels,
            # use_cache=False,
        )
        lm_loss = lm_outputs.loss

        lm_last_hidden_state = lm_outputs.hidden_states[-1]
        num_query_tokens = cell_inputs_llm.shape[1]
        
        if has_cell_output:
            have_gen_mask = torch.nonzero(text_tokens.gen_mask, as_tuple=True)[0].unique()
            # origin_embeds = cell_inputs_llm[have_gen_mask].reshape(-1, lm_emb_dim)
            origin_embeds = cell_inputs_llm[have_gen_mask] # [num_active_samples, num_q, D]

            output_cell_embeds = lm_last_hidden_state[text_tokens.gen_mask] # .reshape(-1, num_query_tokens, lm_emb_dim) # [num_active_samples, num_q, D]
            # print(f"{have_gen_mask=}")
            # num_active_samples = output_cell_embeds.shape[0] // num_query_tokens
            num_active_samples = have_gen_mask.shape[0]
            rec_cell_embeds = output_cell_embeds.view(num_active_samples, -1, lm_emb_dim) # [num_active_samples, num_q, D]
            # rec_cell_embeds = self.decoder_proj(output_cell_embeds) # [num_active_samples, num_q, D]
            # rec_cell_embeds = rec_cell_embeds.reshape(-1, lm_emb_dim) # [num_active_samples x num_q, D]
            rec_loss = F.mse_loss(rec_cell_embeds, origin_embeds.detach())

        else:
            output_cell_embeds = torch.randn(1, num_query_tokens, lm_emb_dim).to(device) + \
                            0.0 * lm_last_hidden_state[0, :num_query_tokens, :]
            rec_loss = 0.0 * output_cell_embeds.sum()

        total_loss = lm_loss + rec_loss

        if self.train_decoder:
            cell_inputs_llm = cell_inputs_llm.detach()
            # print(f"\n===\n{cell_inputs_llm.shape=}")
            decoder_output = self.expr_decoder(cell_inputs_llm[:,:1,:].squeeze(1), gene_inputs['gene_ids'])

            gene_values = gene_inputs['values'].to(torch.float32)
            values1p = process_row(gene_values, 1e2, True, True)
            
            rec_expr_loss = F.mse_loss(decoder_output, values1p.detach())
            # rec_expr_loss = F.mse_loss(decoder_output, gene_values.detach())

        total_loss = total_loss + rec_expr_loss.sum()
        return {'loss': total_loss, 'lm_loss': lm_loss, 'rec_loss': rec_loss, 'expr_loss': rec_expr_loss}

    @torch.no_grad()
    def generate(
        self,
        batch,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # return 'TEXT GENERATION NOT IMPLEMENTED YET'
        
        text_tokens, gene_inputs, texts = batch
        boc_token_id = self.llm_tokenizer.encode(BOC_TOKEN, add_special_tokens=False)[0]
        eoc_token_id = self.llm_tokenizer.encode(EOC_TOKEN, add_special_tokens=False)[0]
        if self.valrank:
            slope = -0.6756886579917628
            intercept = 2.237853249084892
            gene_inputs["values"] = modify_values_by_sorted_rank_tensor(gene_inputs["values"],
                                                                slope=slope,
                                                                intercept=intercept)

        # text_decode = self.llm_tokenizer.batch_decode(text_tokens.input_ids, skip_special_tokens=False)
        # print(f"{text_decode=}")

        # print("=== text_tokens.input_ids ===")
        # print(text_tokens.input_ids)
        # print("=== text_tokens.cmp_mask ===")
        # print(text_tokens.cmp_mask)
        # print("=== text_tokens.gen_mask ===")
        # print(text_tokens.gen_mask)
        inputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids) # [batch_size, seq_len, lm_emb_dim]
        batch_size, seq_len, lm_emb_dim = inputs_embeds.shape

        cell_encoder_output = self.cell_encoder(gene_inputs) # [batch_size, seq_len, d_model]
        if not self.tune_gene_encoder:
            cell_encoder_output = cell_encoder_output.detach()

        
        # print(f"{cell_encoder_output.shape=}")   
        # print(f"{cell_encoder_output[:,:,:16]=}")    

        gene_features = self.ln_gene(cell_encoder_output)


        # print(f"{gene_features.shape=}")   
        # print(f"{gene_features[:,:,:16]=}")    

        gene_padding_mask = gene_inputs['padding_mask'].to(cell_encoder_output.device) # [batch_size, seq_len], 0 if not padding, 1 if padding
        gene_attention_mask = torch.logical_not(gene_padding_mask).long() # [batch_size, seq_len], 1 if not padding, 0 if padding
        
        device = gene_features.device
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=gene_features,
            encoder_attention_mask=gene_attention_mask,
            use_cache=True,
            return_dict=True,
        )

        # print(f"{query_output.last_hidden_state.shape=}")
        # print(f"{query_output.last_hidden_state[:,:,:16]=}")

        cell_inputs_llm = self.llm_proj(query_output.last_hidden_state) # shape = [B, num_q, D]

        # print(f"{cell_inputs_llm.shape=}")
        # print(f"{cell_inputs_llm[:,:,:16]=}")


        """
        atts_llm = torch.ones(cell_inputs_llm.size()[:-1], dtype=torch.long).to(device)
        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100)
        empty_targets = (torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100))
        targets = torch.cat([empty_targets, targets], dim=1)
        attention_mask = torch.cat([atts_llm, text_tokens.attention_mask], dim=1)
        """

        has_cell_input = text_tokens.cmp_mask.sum().item() > 0 # text_tokens.cmp_mask: [batch_size, seq_len]
        has_cell_output = text_tokens.gen_mask.sum().item() > 0 # text_tokens.gen_mask: [batch_size, seq_len]

        
        if has_cell_input:
            if self.use_sentence:
                values = gene_inputs["values"]
                gene_ids = gene_inputs["gene_ids"]
                # 使用topk高效获取前k大值的索引
                num_query_token = cell_inputs_llm.shape[1]  # 动态获取当前batch的num_query_token
                topk_values, topk_indices = torch.topk(values, k=num_query_token, dim=1, sorted=True)
                topk_gene_ids = torch.gather(gene_ids, dim=1, index=topk_indices)
                
                # 转换为token ids并获取嵌入
                token_ids = self.gene_id_to_token_id_tensor[topk_gene_ids]
                gene_token_embeds = self.llm_model.get_input_embeddings()(token_ids)

                have_cmp_mask = torch.nonzero(text_tokens.cmp_mask, as_tuple=True)[0].unique()
                inputs_embeds[text_tokens.cmp_mask] = gene_token_embeds[have_cmp_mask].reshape(-1, lm_emb_dim)
            else:
                have_cmp_mask = torch.nonzero(text_tokens.cmp_mask, as_tuple=True)[0].unique()
                inputs_embeds[text_tokens.cmp_mask] = cell_inputs_llm[have_cmp_mask].reshape(-1, lm_emb_dim)
        # else:
        #     inputs_embeds[:1, :query_output.last_hidden_state.shape[1], :] += 0.0 * cell_inputs_llm[:1, :, :]

        # if has_cell_input: ##
        #     # inputs_embeds[text_tokens.cmp_mask] = cell_inputs_llm.reshape(-1, lm_emb_dim)
        #     have_cmp_mask = torch.nonzero(text_tokens.cmp_mask, as_tuple=True)[0].unique()
            
        #     inputs_embeds[text_tokens.cmp_mask] = cell_inputs_llm[have_cmp_mask].reshape(-1, lm_emb_dim)
        # else:
        #     inputs_embeds[:1, :self.llm_model.config.hidden_size, :] += 0.0 * cell_inputs_llm[:1, :, :]

        # print(f"{inputs_embeds.shape=}")
        # print(f"{inputs_embeds[:,:,:16]=}")

        # print(f"{inputs_embeds[:, (2+32+1):, :].shape=}")
        # print(f"{inputs_embeds[:, (2+32+1):, :16]=}")

        # inputs_embeds = inputs_embeds[:, :(2+32+1), :]



        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=text_tokens.attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            # labels=text_tokens.labels,
            # use_cache=False,
        )

        # print(outputs.hidden_states[-1])

        # generate_ids = outputs.sequences[ : , inputs_embeds.shape[1]:]
        generate_ids = outputs.sequences
        text_mask = torch.ones_like(generate_ids, dtype=torch.bool)

        # eoc_indices = torch.where(generate_ids == eoc_token_id)[0].tolist()[0]

        last_hidden_state = outputs.hidden_states[-1]

        # print(f"{last_hidden_state.shape=}")
        # print(f"{text_tokens.gen_mask.shape=}")
        # print(f"{cell_inputs_llm.shape=}")
        
        # if has_cell_output:
        #     num_query_tokens = cell_inputs_llm.shape[1]
        #     target_embeds = inputs_embeds[text_tokens.gen_mask] # [num_active_samples x num_q, D]
        #     output_cell_embeds = last_hidden_state[ : , text_tokens.gen_mask, :].reshape(-1, num_query_tokens, lm_emb_dim) # [num_active_samples, num_q, D]
        #     rec_cell_embeds = self.decoder_proj(output_cell_embeds) # [num_active_samples, num_q, D]
        #     rec_cell_embeds = rec_cell_embeds.reshape(-1, lm_emb_dim) # [num_active_samples x num_q, D]

        # else:
        #     rec_cell_embeds = None

        # print(f"{generate_ids=}")

        # text_mask[generate_ids == boc_token_id] = False
        # generate_ids = generate_ids[text_mask]
        generate_text = self.llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        # print(f"{generate_text=}")
        
        if self.train_decoder:
            cell_inputs_llm = cell_inputs_llm.detach()
            # print(f"\n===\n{cell_inputs_llm.shape=}")
            decoder_output = self.expr_decoder(cell_inputs_llm[:,:1,:].squeeze(1), gene_inputs['gene_ids'])
            # print(f"{decoder_output.shape=}")
            # print(f"{gene_inputs['values']=}")
            # print(f"{decoder_output['pred'].shape=}")
            # print(f"{gene_inputs['values'].shape=}")
            # print(gene_inputs['values'])

            # gene_values = gene_inputs['values'].to(torch.float32)
            # values1p = gene_values.clone()
            # values1p[:, 1:] = torch.log1p(gene_values[:, 1:])
            # # values1p = torch.log1p(gene_values)
            # # print("values1p:", values1p.min().item(), values1p.max().item())
            # rec_expr_loss = F.mse_loss(decoder_output, values1p.detach())

        return_dict = {}

        return_dict['text'] = generate_text
        return_dict['values1p'] = decoder_output

        # pprint(return_dict)

        return return_dict
        # {
        #     'text': generate_text,
        #     # 'has_img_output': has_cell_output,
        #     # 'cell_gen_feat': rec_cell_embeds,
        # }
    

class GeneExpressionModel(nn.Module):
    def __init__(self, num_genes, gene_embed_dim, cell_feat_dim, hidden_dim):
        super().__init__()
        # 将Cell Features映射为Key-Value
        self.cell_proj = nn.Linear(cell_feat_dim, 2*hidden_dim)
        # self.norm = nn.LayerNorm(hidden_dim)  # 添加归一化层
        self.hidden_dim = hidden_dim
        # 基因ID Embedding层
        self.gene_embed = nn.Embedding(num_genes, gene_embed_dim)
        # Cross Attention模块
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=False)  # 注意batch_first=False
        # 预测表达值的MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + gene_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, cell_feats, gene_ids):
        # cell_feats: [batch_size, cell_feat_dim]
        # gene_ids: [batch_size, num_genes]
        batch_size, num_genes = gene_ids.shape
        
        # 生成基因Embeddings [batch_size, num_genes, gene_embed_dim]
        gene_embeds = self.gene_embed(gene_ids)  # shape: [batch_size, seq_len, embed_dim]
        # print("gene_embeds输出范围:", gene_embeds.min().item(), gene_embeds.max().item())
        # 投影Cell Features为Key-Value
        cell_kv = self.cell_proj(cell_feats)  # shape: [batch_size, 2*hidden_dim]
        cell_kv = cell_kv.view(batch_size, 1, 2*self.hidden_dim)  # [batch_size, 1, 2*hidden_dim]
        cell_k, cell_v = torch.split(cell_kv, self.hidden_dim, dim=-1)  # [batch_size, 1, hidden_dim]
        
        # 调整维度顺序为 (seq_len, batch_size, embed_dim)
        query = gene_embeds.permute(1,0,2)  # [num_genes, batch_size, embed_dim]
        key = cell_k.permute(1,0,2)         # [1, batch_size, hidden_dim]
        value = cell_v.permute(1,0,2)       # [1, batch_size, hidden_dim]
        # print("query:", query.min().item(), query.max().item())
        # print("key:", key.min().item(), key.max().item())
        # print("value:", value.min().item(), value.max().item())
        # print(f"{query.shape=}")
        # print(f"{key.shape=}")
        # print(f"{value.shape=}")
        # Cross Attention计算
        context, _ = self.cross_attn(
            query=query,  # [num_genes, batch_size, hidden_dim]
            key=key,      # [1, batch_size, hidden_dim]
            value=value   # [1, batch_size, hidden_dim]
        )
        # context = self.norm(context)  # 归一化
        context = context.permute(1,0,2)  # 恢复为 [batch_size, num_genes, hidden_dim]
        
        # 拼接上下文与基因Embedding，预测表达值
        combined = torch.cat([context, gene_embeds], dim=-1)  # [batch_size, num_genes, hidden_dim + gene_embed_dim]
        expr = self.mlp(combined).squeeze(-1)  # [batch_size, num_genes]
        return expr