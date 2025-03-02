import torch
from torch_geometric.data import Dataset
from pathlib import Path
import random
import os
from .pretrain_dataset import load_all_counts
from .data_utils import process_adata_dir, CellTextHelper
import json
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformers import BertTokenizer
from scgpt.tokenizer import pad_batch
from torchtext.vocab import Vocab
import numpy as np

def pad_batch_hvalue(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    gene_ids_list = []
    values_list = []
    mod_types_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]
        # if len(gene_ids) > max_len:
        #     # sample max_len genes
        #     if not cls_appended:
        #         idx = np.random.choice(len(gene_ids), max_len, replace=False)
        #     else:
        #         idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
        #         idx = idx + 1
        #         idx = np.insert(idx, 0, 0)
        #     gene_ids = gene_ids[idx]
        #     values = values[idx]
        #     if mod_types is not None:
        #         mod_types = mod_types[idx]
        if len(gene_ids) > max_len:
            # 按 value 值排序并选择最大的 max_len 个索引
            if not cls_appended:
                idx = np.argsort(values)[-max_len:]
            else:
                idx = np.argsort(values[1:])[-(max_len - 1):] + 1
                idx = np.insert(idx, 0, 0)  # 确保第一个元素包含在索引中
            
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]

        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )
        
        # values = process_row(values, 100, True, cls_appended)

        gene_ids_list.append(gene_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded


def process_row(
    row,
    normalize_total: Optional[float] = None,
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
        head = processed[:1]  # 保持第一个元素不变
        tail = processed[1:]
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
        return torch.cat([head, tail])
    return tail

class ClsCollater:
    def __init__(self, text_tokenizer, text_max_len, cell_max_len, vocab):
        self.text_tokenizer = text_tokenizer
        self.text_max_len = text_max_len
        self.cell_max_len = cell_max_len
        self.vocab = vocab
        self.pad_token = "<pad>"
        self.pad_value = 0

    def __call__(self, batch):
        cell_info = batch
        cell_batch = [(x["gene_ids"], x["values"], None) for x in cell_info]
        batch_padded = pad_batch_hvalue(
            cell_batch,
            max_len=self.cell_max_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            cls_appended=True,
        )
        src_key_padding_mask = batch_padded["genes"].eq(self.vocab[self.pad_token])
        gene_tokens = {
            "gene_ids": batch_padded["genes"], # [batch_size, seq_len]
            "values": batch_padded["values"], # [batch_size, seq_len]
            'padding_mask': src_key_padding_mask, # [batch_size, seq_len]
        }

        labels = [x['label'] for x in cell_info]
        labels = torch.Tensor(labels).long()

        return gene_tokens, labels

class ClsDataset(Dataset):
    def __init__(self, h5ad_path, text_max_len, vocab, bert_model_name):
        super(ClsDataset, self).__init__()
        self.h5ad_path = h5ad_path
        h5ad_file = Path(h5ad_path)
        root = h5ad_file.parent
        self.vocab = vocab
        filename = h5ad_file.stem
        self.bert_model_name = bert_model_name

        assert h5ad_file.exists()
        print(h5ad_file)
        print(root)
        # single source of data
        processed_path = root / f'{filename}_databanks'
        if not processed_path.exists():
            process_adata_dir(h5ad_path, processed_path, vocab=self.vocab, token_col='gene_symbols', label_col='str_labels')
        else:
            print(f"load from {processed_path.as_posix()}")
            # process_adata_dir(h5ad_path, processed_path, vocab=self.vocab, token_col='gene_symbols', label_col='str_labels')
        self.dataset = load_all_counts(processed_path / 'all_counts')
        
        if (root/ 'type2text.json').exists():
            with open(root / 'type2text.json', 'r') as f:
                self.type2text = json.load(f)
            self.type2index = {k.lower(): i for i, k in enumerate(self.type2text.keys())}
        self.text_max_len = text_max_len
        if bert_model_name == "scibert":
            bert_name = 'allenai/scibert_scivocab_uncased'
            # print("bert load scibert as tokenizer")
        elif bert_model_name == 'pubmedbert':
            bert_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
            # print("bert load pubmedbert as tokenizer")
        else:
            bert_name = 'bert_pretrained/'
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)

        self.dataset = self.dataset.shuffle(seed=42)
        # self.dataset = self.dataset.select(range(len(self.dataset)//100))

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.dataset)
    
    def get_text_list(self):
        text_list = [ f'This cell belongs to {k}: {self.type2text[k]}'
            for k in self.type2text.keys()]
        # print("\n\n@@@\n\n@@@\n\n@@@\n\n")
        # with open("temp_text_list.txt", "w") as ff:
        #     for ii in text_list:
        #         ff.write(ii)
        text_tokens, attention_masks = self.tokenizer_text(text_list)
        return text_tokens, attention_masks

    def __getitem__(self, index):
        gene_ids = torch.tensor([self.vocab["<cls>"]] + self.dataset[index]['genes'], dtype=torch.long)
        values = torch.tensor([-2] + self.dataset[index]['expressions'], dtype=torch.float)
        str_label = self.dataset[index]['str_labels']
        label = torch.tensor(self.type2index[str_label.lower()], dtype=torch.long) # TODO: add label
        gene_inputs = {
            'gene_ids': gene_ids,
            'values': values,
            'label': label
        }
        # text = self.metadata_to_text(self.dataset[index]['metadata'])
        return gene_inputs

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask
    

import random
import numpy as np
from collections import defaultdict

class H5Dataset(Dataset):
    def __init__(self, h5ad_path, text_max_len, vocab, bert_model_name, ratio=1.0, debug_ratio=-1, equal_num=0):
        super(H5Dataset, self).__init__()
        self.h5ad_path = h5ad_path
        h5ad_file = Path(h5ad_path)
        root = h5ad_file.parent
        self.vocab = vocab
        filename = h5ad_file.stem
        self.bert_model_name = bert_model_name
        self.equal_num = equal_num  # 保存equal_num参数

        assert h5ad_file.exists()
        print(h5ad_file)
        print(root)
        # single source of data
        processed_path = root / f'{filename}_databanks'
        if not processed_path.exists():
            process_adata_dir(h5ad_path, processed_path, vocab=self.vocab, token_col='gene_symbols', label_col='str_labels')
        else:
            print(f"load from {processed_path.as_posix()}")

        self.dataset = load_all_counts(processed_path / 'all_counts')
        
        if (root/ 'type2text.json').exists():
            with open(root / 'type2text.json', 'r') as f:
                self.type2text = json.load(f)
            self.type2index = {k.lower(): i for i, k in enumerate(self.type2text.keys())}
        self.text_max_len = text_max_len
        if bert_model_name == "scibert":
            bert_name = 'allenai/scibert_scivocab_uncased'
        elif bert_model_name == 'pubmedbert':
            bert_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
        else:
            bert_name = 'bert_pretrained/'
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)

        self.cell_text_helper = CellTextHelper()

        self.dataset = self.dataset.shuffle(seed=42)
        if ratio > 0:
            self.dataset = self.dataset.select(range(int(len(self.dataset)*ratio)))
        else:
            self.dataset = self.dataset.select(range(int(len(self.dataset)*(-ratio)), len(self.dataset)))

        if debug_ratio > 0:
            sampled_indices = random.sample(range(len(self.dataset)), len(self.dataset)//debug_ratio)
            self.dataset = self.dataset.select(sampled_indices)

        # 处理equal_num参数
        if self.equal_num > 0:
            label_to_indices = defaultdict(list)
            # 收集每个类别的索引
            for idx in range(len(self.dataset)):
                str_label = self.dataset[idx]['str_labels']
                label_to_indices[str_label].append(idx)
            # 对每个类别进行有放回采样
            np.random.seed(42)  # 固定随机种子保证可重复性
            sampled_indices = []
            for indices in label_to_indices.values():
                # 使用有放回采样，确保每个类别有equal_num个样本
                replacements = len(indices) < self.equal_num
                sampled = np.random.choice(indices, size=self.equal_num, replace=True)
                sampled_indices.extend(sampled.tolist())
            # 打乱所有样本的顺序
            np.random.shuffle(sampled_indices)
            self.dataset = self.dataset.select(sampled_indices)

    # 其余方法保持不变
    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.dataset)
    
    def get_text_list(self):
        text_list = [ f'This cell belongs to {k}: {self.type2text[k]}'
            for k in self.type2text.keys()]
        text_tokens, attention_masks = self.tokenizer_text(text_list)
        return text_tokens, attention_masks

    def __getitem__(self, index):
        gene_ids = torch.tensor([self.vocab["<cls>"]] + self.dataset[index]['genes'], dtype=torch.long)
        values = torch.tensor([-2] + self.dataset[index]['expressions'], dtype=torch.float)
        str_label = self.dataset[index]['str_labels']
        label = torch.tensor(self.type2index[str_label.lower()], dtype=torch.long)
        gene_inputs = {
            'gene_ids': gene_ids,
            'values': values,
            'label': label,
            "cell_type":str_label
        }
        text = self.cell_text_helper.cellxgene_metadata_to_text({"cell_type":str_label})
        return gene_inputs, text

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


class H5Dataset_old(Dataset):
    def __init__(self, h5ad_path, text_max_len, vocab, bert_model_name, ratio=1.0, debug_ratio=-1):
        super(H5Dataset_old, self).__init__()
        self.h5ad_path = h5ad_path
        h5ad_file = Path(h5ad_path)
        root = h5ad_file.parent
        self.vocab = vocab
        filename = h5ad_file.stem
        self.bert_model_name = bert_model_name

        assert h5ad_file.exists()
        print(h5ad_file)
        print(root)
        # single source of data
        processed_path = root / f'{filename}_databanks'
        if not processed_path.exists():
            process_adata_dir(h5ad_path, processed_path, vocab=self.vocab, token_col='gene_symbols', label_col='str_labels')
        else:
            print(f"load from {processed_path.as_posix()}")
            # process_adata_dir(h5ad_path, processed_path, vocab=self.vocab, token_col='gene_symbols', label_col='str_labels')

        self.dataset = load_all_counts(processed_path / 'all_counts')
        
        if (root/ 'type2text.json').exists():
            with open(root / 'type2text.json', 'r') as f:
                self.type2text = json.load(f)
            self.type2index = {k.lower(): i for i, k in enumerate(self.type2text.keys())}
        self.text_max_len = text_max_len
        if bert_model_name == "scibert":
            bert_name = 'allenai/scibert_scivocab_uncased'
            # print("bert load scibert as tokenizer")
        elif bert_model_name == 'pubmedbert':
            bert_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
            # print("bert load pubmedbert as tokenizer")
        else:
            bert_name = 'bert_pretrained/'
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)

        self.cell_text_helper = CellTextHelper()

        self.dataset = self.dataset.shuffle(seed=42)
        if ratio > 0:
            self.dataset = self.dataset.select(range(int(len(self.dataset)*ratio)))
        else:
            self.dataset = self.dataset.select(range(int(len(self.dataset)*(-ratio)), len(self.dataset)))

        if debug_ratio > 0:
            sampled_indices = random.sample(range(len(self.dataset)), len(self.dataset)//debug_ratio)
            self.dataset = self.dataset.select(sampled_indices)

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.dataset)
    
    def get_text_list(self):
        text_list = [ f'This cell belongs to {k}: {self.type2text[k]}'
            for k in self.type2text.keys()]
        # print("\n\n@@@\n\n@@@\n\n@@@\n\n")
        # with open("temp_text_list.txt", "w") as ff:
        #     for ii in text_list:
        #         ff.write(ii)
        text_tokens, attention_masks = self.tokenizer_text(text_list)
        return text_tokens, attention_masks

    def __getitem__(self, index):
        gene_ids = torch.tensor([self.vocab["<cls>"]] + self.dataset[index]['genes'], dtype=torch.long)
        values = torch.tensor([-2] + self.dataset[index]['expressions'], dtype=torch.float)
        str_label = self.dataset[index]['str_labels']
        label = torch.tensor(self.type2index[str_label.lower()], dtype=torch.long) # TODO: add label
        gene_inputs = {
            'gene_ids': gene_ids,
            'values': values,
            'label': label,
            "cell_type":str_label
        }
        text = self.cell_text_helper.cellxgene_metadata_to_text({"cell_type":str_label})
        return gene_inputs, text

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


class ClsDataset_fewshot(Dataset):
    def __init__(self, h5ad_path, text_max_len, vocab, bert_model_name):
        super(ClsDataset_fewshot, self).__init__()
        self.h5ad_path = h5ad_path
        h5ad_file = Path(h5ad_path)
        root = h5ad_file.parent
        self.vocab = vocab
        filename = h5ad_file.stem
        self.bert_model_name = bert_model_name

        assert h5ad_file.exists()
        print(h5ad_file)
        print(root)
        # single source of data
        processed_path = root / f'{filename}_databanks'
        if not processed_path.exists():
            process_adata_dir(h5ad_path, processed_path, vocab=self.vocab, token_col='gene_symbols', label_col='str_labels')
        else:
            print(f"load from {processed_path.as_posix()}")
            # process_adata_dir(h5ad_path, processed_path, vocab=self.vocab, token_col='gene_symbols', label_col='str_labels')
        self.dataset = load_all_counts(processed_path / 'all_counts')
        
        if (root/ 'type2text.json').exists():
            with open(root / 'type2text.json', 'r') as f:
                self.type2text = json.load(f)
            self.type2index = {k.lower(): i for i, k in enumerate(self.type2text.keys())}
        self.text_max_len = text_max_len
        if bert_model_name == "scibert":
            bert_name = 'allenai/scibert_scivocab_uncased'
            # print("bert load scibert as tokenizer")
        elif bert_model_name == 'pubmedbert':
            bert_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
            # print("bert load pubmedbert as tokenizer")
        else:
            bert_name = 'bert_pretrained/'
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)

        self.dataset = self.dataset.shuffle(seed=42)
        # self.dataset = self.dataset.select(range(len(self.dataset)//100))

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.dataset)
    
    def get_text_list(self):
        text_list = [ f'This cell belongs to {k}: {self.type2text[k]}'
            for k in self.type2text.keys()]
        # print("\n\n@@@\n\n@@@\n\n@@@\n\n")
        # with open("temp_text_list.txt", "w") as ff:
        #     for ii in text_list:
        #         ff.write(ii)
        text_tokens, attention_masks = self.tokenizer_text(text_list)
        return text_tokens, attention_masks

    def __getitem__(self, index):
        gene_ids = torch.tensor([self.vocab["<cls>"]] + self.dataset[index]['genes'], dtype=torch.long)
        values = torch.tensor([-2] + self.dataset[index]['expressions'], dtype=torch.float)
        str_label = self.dataset[index]['str_labels']
        label = torch.tensor(self.type2index[str_label.lower()], dtype=torch.long) # TODO: add label
        gene_inputs = {
            'gene_ids': gene_ids,
            'values': values,
            'label': label
        }
        # text = self.metadata_to_text(self.dataset[index]['metadata'])
        return gene_inputs

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask
    
    def get_fewshot_dataset(self, n):
        # for every label, select n sample, return dataset as fewshot training dataset
        # remove these sample from the dataset
        fewshot_data = []
        remaining_data = []
        label_count = {label: 0 for label in self.type2index.values()}

        for data in self.dataset:
            str_label = data['str_labels'].lower()
            label = self.type2index[str_label]
            if label_count[label] < n:
                fewshot_data.append(data)
                label_count[label] += 1
            else:
                remaining_data.append(data)

        self.dataset = remaining_data
        return fewshot_data