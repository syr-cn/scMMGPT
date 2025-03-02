# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from data_provider.pretrain_dataset import Stage1Dataset, Stage1Dataset_tabula
from data_provider.retrieval_dataset import ClsDataset, ClsCollater, ClsDataset_fewshot
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import json
from pathlib import Path
from scgpt.tokenizer import pad_batch
from typing import Dict, Iterable, List, Optional, Tuple, Union
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab


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


import torch

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


class Stage1Collater:
    def __init__(self, text_tokenizer, text_max_len, cell_max_len, vocab):
        self.text_tokenizer = text_tokenizer
        self.text_max_len = text_max_len
        self.cell_max_len = cell_max_len
        self.vocab = vocab
        self.pad_token = "<pad>"
        self.pad_value = 0

    def __call__(self, batch):
        cell_info, texts = zip(*batch)
        
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

        self.text_tokenizer.padding_side = 'right'
        text_tokens = self.text_tokenizer(
            text=texts,
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            max_length=self.text_max_len,
            return_tensors='pt',
            return_attention_mask=True
        )
        return text_tokens, gene_tokens

class Stage1DM(LightningDataModule):
    def __init__(
        self,
        root,
        zeroshot_cls_datafiles,
        zeroshot_retrieval_datafiles,
        vocab,
        num_workers: int = 0,
        batch_size: int = 256,
        match_batch_size: int = 64,
        text_max_len: int = 128,
        cell_max_len: int = 128,
        bert_name: str='scibert',
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = match_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.cell_max_len = cell_max_len
        
        if args.train_on_tabula_ratio>0:
            assert args.tabula_path is not None
            assert args.tabula_train_indices_path is not None
            assert args.tabula_test_indices_path is not None

            with open(args.tabula_train_indices_path, 'r') as f:
                train_indices = json.load(f)
            with open(args.tabula_test_indices_path, 'r') as f:
                test_indices = json.load(f)
            self.train_dataset = Stage1Dataset_tabula(Path(args.tabula_path), vocab, debug_ratio=10, from_disk=False, sampled_indices=train_indices, training_ratio=args.train_on_tabula_ratio)
            self.val_dataset = Stage1Dataset_tabula(Path(args.tabula_path), vocab, debug_ratio=0, from_disk=False, sampled_indices=test_indices, training_ratio=args.train_on_tabula_ratio)
        else:
            self.train_dataset = Stage1Dataset(root, vocab, debug_ratio=-1, from_padding=True, test_split_num=1000)
            self.val_dataset = Stage1Dataset(root, vocab, debug_ratio=1000, from_padding=True)
        
        self.vocab = vocab
        print('len(train_dataset)', len(self.train_dataset))
        print('len(val_dataset)', len(self.val_dataset))
        
        if zeroshot_cls_datafiles:
            self.zeroshot_cls_datasets = [ClsDataset(h5ad_path, text_max_len, vocab, bert_name) for h5ad_path in zeroshot_cls_datafiles]
            self.zeroshot_cls_loaders = { Path(dataset_match.h5ad_path).as_posix() :
                DataLoader(dataset_match, 
                    batch_size=self.match_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers, 
                    pin_memory=False, 
                    drop_last=True, 
                    persistent_workers=True,
                    collate_fn=ClsCollater(
                        dataset_match.tokenizer,
                        text_max_len,
                        cell_max_len,
                        vocab,
                    )
                )
                for dataset_match in self.zeroshot_cls_datasets
            }
        else:
            self.zeroshot_cls_datasets = []
            self.zeroshot_cls_loaders = {}

        if args is not None and args.test_on_tabula:
            if args.tabula_test_indices_path is not None:
                with open(args.tabula_test_indices_path, 'r') as f:
                    test_indices = json.load(f)
                self.val_dataset_tabula = Stage1Dataset_tabula(Path(args.tabula_path),
                                                        vocab, debug_ratio=0, from_disk=False, sampled_indices=test_indices)
                self.val_dataset_tabula.subset_types = [type_id for type_id in self.val_dataset_tabula.type2index.values() if type_id not in self.train_dataset.subset_types]
                print(f'Loaded {len(self.val_dataset_tabula.subset_types)} types for the test set.')
                del self.val_dataset_tabula.subset_types
                print('self.val_dataset_tabula.subset_types is deleted.')
            else:
                self.val_dataset_tabula = Stage1Dataset_tabula(Path(args.tabula_path), 
                                                        vocab, debug_ratio=30, from_disk=False)

            self.zeroshot_cls_loaders[Path(args.tabula_path).as_posix()] = DataLoader(
                                                                            self.val_dataset_tabula,
                                                                            batch_size=args.tabula_batchsize,
                                                                            shuffle=False,
                                                                            num_workers=self.num_workers,
                                                                            pin_memory=False,
                                                                            drop_last=True,
                                                                            persistent_workers=True,
                                                                            collate_fn=ClsCollater(
                                                                                self.val_dataset_tabula.tokenizer,
                                                                                self.text_max_len,
                                                                                self.cell_max_len,
                                                                                self.vocab,
                                                                            ),
                                                                        )
            
            print('len(val_dataset_tabula)', len(self.val_dataset_tabula))
            print(self.zeroshot_cls_loaders)
        
        if zeroshot_retrieval_datafiles:
            self.zeroshot_retrieval_datasets = [ClsDataset(h5ad_path, text_max_len, vocab, bert_name) for h5ad_path in zeroshot_retrieval_datafiles]
            self.zeroshot_retrieval_loaders = [
                DataLoader(dataset_match, 
                    batch_size=self.match_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers, 
                    pin_memory=False, 
                    drop_last=False, 
                    persistent_workers=True)
                for dataset_match in self.zeroshot_retrieval_datasets
            ]
        else:
            self.zeroshot_retrieval_datasets = []
            self.zeroshot_retrieval_loaders = []

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=Stage1Collater(
                self.train_dataset.tokenizer,
                self.text_max_len,
                self.cell_max_len,
                self.vocab,
            ),
        )
        # print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True, # change for contrast_batch bug
            persistent_workers=True,
            collate_fn=Stage1Collater(
                self.val_dataset.tokenizer,
                self.text_max_len,
                self.cell_max_len,
                self.vocab,
            ),
        )

        return loader
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--graph_aug', type=str, default='dnodes')
        parser.add_argument('--zeroshot_cls_datafiles', default=[], type=lambda x: [i.strip() for i in x.split(',') if i.strip()])
        parser.add_argument('--zeroshot_retrieval_datafiles', default=[], type=lambda x: [i.strip() for i in x.split(',') if i.strip()])
        parser.add_argument('--text_aug', action='store_true', default=False)
        parser.add_argument('--use_phy_eval', action='store_true', default=False)
        parser.add_argument('--train_on_tabula_ratio',  type=float, default=-1)
        parser.add_argument('--test_on_tabula', action='store_true', default=False)
        parser.add_argument('--tabula_path', type=str, default=None)
        parser.add_argument('--tabula_train_indices_path', type=str, default=None)
        parser.add_argument('--tabula_test_indices_path', type=str, default=None)
        parser.add_argument('--tabula_batchsize', type=int, default=3)
        return parent_parser
    

class Stage1DM_fewshot(LightningDataModule):
    def __init__(
        self,
        root,
        zeroshot_cls_datafiles,
        zeroshot_retrieval_datafiles,
        vocab,
        num_workers: int = 0,
        batch_size: int = 256,
        match_batch_size: int = 64,
        text_max_len: int = 128,
        cell_max_len: int = 128,
        bert_name: str='scibert',
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = match_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.cell_max_len = cell_max_len
        
        # self.train_dataset = Stage1Dataset(root, vocab, debug_ratio=-1, from_padding=True)
        # self.val_dataset = Stage1Dataset(root, vocab, debug_ratio=1000, from_padding=True)
        
        # self.vocab = vocab
        # print('len(train_dataset)', len(self.train_dataset))
        # print('len(val_dataset)', len(self.val_dataset))
        
        if zeroshot_cls_datafiles:
            self.zeroshot_cls_datasets = [ClsDataset_fewshot(h5ad_path, text_max_len, vocab, bert_name) for h5ad_path in zeroshot_cls_datafiles]
            self.zeroshot_cls_loaders = { Path(dataset_match.h5ad_path).as_posix() :
                DataLoader(dataset_match, 
                    batch_size=self.match_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers, 
                    pin_memory=False, 
                    drop_last=True, 
                    persistent_workers=True,
                    collate_fn=ClsCollater(
                        dataset_match.tokenizer,
                        text_max_len,
                        cell_max_len,
                        vocab,
                    )
                )
                for dataset_match in self.zeroshot_cls_datasets
            }
        else:
            self.zeroshot_cls_datasets = []
            self.zeroshot_cls_loaders = {}

        if args is not None and args.test_on_tabula:
            self.val_dataset_tabula = Stage1Dataset_tabula(Path(args.tabula_path), 
                                                        vocab, debug_ratio=10, from_disk=False)

            self.zeroshot_cls_loaders[Path(args.tabula_path).as_posix()] = DataLoader(
                                                                            self.val_dataset_tabula,
                                                                            batch_size=args.tabula_batchsize,
                                                                            shuffle=False,
                                                                            num_workers=self.num_workers,
                                                                            pin_memory=False,
                                                                            drop_last=True,
                                                                            persistent_workers=True,
                                                                            collate_fn=ClsCollater(
                                                                                self.val_dataset_tabula.tokenizer,
                                                                                self.text_max_len,
                                                                                self.cell_max_len,
                                                                                self.vocab,
                                                                            ),
                                                                        )
            
            print('len(val_dataset_tabula)', len(self.val_dataset_tabula))
            print(self.zeroshot_cls_loaders)
        
        if zeroshot_retrieval_datafiles:
            self.zeroshot_retrieval_datasets = [ClsDataset(h5ad_path, text_max_len, vocab, bert_name) for h5ad_path in zeroshot_retrieval_datafiles]
            self.zeroshot_retrieval_loaders = [
                DataLoader(dataset_match, 
                    batch_size=self.match_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers, 
                    pin_memory=False, 
                    drop_last=False, 
                    persistent_workers=True)
                for dataset_match in self.zeroshot_retrieval_datasets
            ]
        else:
            self.zeroshot_retrieval_datasets = []
            self.zeroshot_retrieval_loaders = []

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=Stage1Collater(
                self.train_dataset.tokenizer,
                self.text_max_len,
                self.cell_max_len,
                self.vocab,
            ),
        )
        # print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True, # change for contrast_batch bug
            persistent_workers=True,
            collate_fn=Stage1Collater(
                self.val_dataset.tokenizer,
                self.text_max_len,
                self.cell_max_len,
                self.vocab,
            ),
        )

        return loader
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--graph_aug', type=str, default='dnodes')
        parser.add_argument('--zeroshot_cls_datafiles', default=[], type=lambda x: [i.strip() for i in x.split(',') if i.strip()])
        parser.add_argument('--zeroshot_retrieval_datafiles', default=[], type=lambda x: [i.strip() for i in x.split(',') if i.strip()])
        parser.add_argument('--text_aug', action='store_true', default=False)
        parser.add_argument('--use_phy_eval', action='store_true', default=False)
        parser.add_argument('--test_on_tabula', action='store_true', default=False)
        parser.add_argument('--tabula_path', type=str, default=None)
        parser.add_argument('--tabula_batchsize', type=int, default=3)
        return parent_parser