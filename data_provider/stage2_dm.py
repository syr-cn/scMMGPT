# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data_provider.pretrain_dataset import Stage1Dataset
from data_provider.retrieval_dataset import H5Dataset
from data_provider.stage1_dm import pad_batch_hvalue
import re
import numpy as np
import random
from transformers import BatchEncoding


BOC_TOKEN = '<cell>'
CELL_TOKEN = '<cell_{:05d}>'
EOC_TOKEN = '</cell>'


def encode_caption_input_ids(caption_ids, 
                             tokenizer, 
                             max_length, 
                             device, 
                             img_first_flag=None, 
                             num_cell_in_tokens=32, 
                             num_cell_out_tokens=32,
                             fortest=False):
    if img_first_flag is None:
        img_first_flag = np.random.uniform(0, 1) < 0.5

    if len(caption_ids) + num_cell_out_tokens + 4 > max_length:
        img_first_flag = True

    if img_first_flag:
        caption_labels = caption_ids
        cell_tokens = BOC_TOKEN + ''.join([CELL_TOKEN.format(int(item)) for item in range(num_cell_in_tokens)]) + EOC_TOKEN

        cell_ids = tokenizer.encode(cell_tokens, add_special_tokens=False)
        cell_labels = [-100] * len(cell_ids)

        input_ids = [tokenizer.bos_token_id] + cell_ids + caption_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + cell_labels + caption_labels + [tokenizer.eos_token_id]
        ids_gen_mask = [False] * len(input_ids)
        ids_cmp_mask = [False] + [False] + [True] * num_cell_in_tokens + [False] + [False] * len(caption_ids) + [False]

    else:
        caption_labels = [-100] * len(caption_ids)
        cell_tokens = BOC_TOKEN + ''.join([CELL_TOKEN.format(int(item)) for item in range(num_cell_out_tokens)]) + EOC_TOKEN

        cell_ids = tokenizer.encode(cell_tokens, add_special_tokens=False)
        cell_labels = [cell_ids[0]] + [-100] * (len(cell_ids) - 1)

        input_ids = [tokenizer.bos_token_id] + caption_ids + cell_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + caption_labels + cell_labels + [tokenizer.eos_token_id]
        ids_gen_mask = [False] + [False] * len(caption_ids) + [False] + [True] * num_cell_out_tokens + [False] + [False]
        ids_cmp_mask = [False] * len(input_ids)


    if fortest:
        if img_first_flag:
            input_len = len(cell_ids) + 1
            input_ids = input_ids[:input_len]
            attention_mask = attention_mask[:input_len]
            labels = labels[:input_len]
            ids_gen_mask = ids_gen_mask[:input_len]
            ids_cmp_mask = ids_cmp_mask[:input_len]
        else:
            input_len = len(caption_ids) + 1
            input_ids = input_ids[:input_len]
            attention_mask = attention_mask[:input_len]
            labels = labels[:input_len]
            ids_gen_mask = ids_gen_mask[:input_len]
            ids_cmp_mask = ids_cmp_mask[:input_len]


    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        ids_gen_mask = ids_gen_mask[:max_length]
        ids_cmp_mask = ids_cmp_mask[:max_length]

    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool, device=device)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool, device=device)

    # print(f"{input_ids.shape=}")
    # print(f"{attention_mask.shape=}")
    # print(f"{labels.shape=}")
    # print(f"{ids_gen_mask.shape=}")
    # print(f"{ids_cmp_mask.shape=}")
    # print(f"==================")

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'gen_mask': ids_gen_mask,
        'cmp_mask': ids_cmp_mask,
    }

class TrainCollater:
    def __init__(self, text_tokenizer, text_max_len, cell_max_len, vocab, num_query_tokens=32, img_first_flag=None):
        self.text_tokenizer = text_tokenizer
        self.text_max_len = text_max_len
        self.cell_max_len = cell_max_len
        self.vocab = vocab
        self.pad_token = "<pad>"
        self.pad_value = 0
        self.num_query_tokens = num_query_tokens
        self.img_first_flag = img_first_flag

    def __call__(self, batch):
        cell_info, texts = zip(*batch)

        # print(f"####### {texts=}", flush=True)
        
        # assert(False)

        cell_batch = [(x["gene_ids"], x["values"], None) for x in cell_info]
        cell_type_list = [x['cell_type'] for x in cell_info]
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
            "cell_type": cell_type_list # [batch_size, ]
        }
        self.text_tokenizer.padding_side = 'right'
        
        caption_ids_list = [self.text_tokenizer.encode(text, add_special_tokens=False) for text in texts]
        max_ids_len = max(len(ids) for ids in caption_ids_list)
        max_len = min(max_ids_len + self.num_query_tokens + 4, self.text_max_len)
        
        tokenized_texts = [
            encode_caption_input_ids(
                caption_ids,
                self.text_tokenizer,
                max_len,
                device=gene_tokens['gene_ids'].device,
                img_first_flag=self.img_first_flag,
                num_cell_in_tokens=self.num_query_tokens,
                num_cell_out_tokens=self.num_query_tokens,
            )
            for caption_ids in caption_ids_list
        ]

        text_tokens = {
            key: torch.stack([x[key] for x in tokenized_texts]) for key in tokenized_texts[0].keys()
        }
        text_tokens = BatchEncoding(data=text_tokens, tensor_type='pt')

        return text_tokens, gene_tokens


class TestCollater:
    def __init__(self, text_tokenizer, text_max_len, cell_max_len, vocab, num_query_tokens=32, img_first_flag=None):
        self.text_tokenizer = text_tokenizer
        self.text_max_len = text_max_len
        self.cell_max_len = cell_max_len
        self.vocab = vocab
        self.pad_token = "<pad>"
        self.pad_value = 0
        self.num_query_tokens = num_query_tokens
        self.img_first_flag = img_first_flag

    def __call__(self, batch):
        cell_info, texts = zip(*batch)

        # print(f"####### {texts=}", flush=True)
        
        # assert(False)

        cell_batch = [(x["gene_ids"], x["values"], None) for x in cell_info]
        cell_type_list = [x['cell_type'] for x in cell_info]
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
            "cell_type": cell_type_list # [batch_size, ]
        }
        self.text_tokenizer.padding_side = 'right'
        
        caption_ids_list = [self.text_tokenizer.encode(text, add_special_tokens=False) for text in texts]
        max_ids_len = max(len(ids) for ids in caption_ids_list)
        max_len = min(max_ids_len + self.num_query_tokens + 4, self.text_max_len)
        
        tokenized_texts = [
            encode_caption_input_ids(
                caption_ids,
                self.text_tokenizer,
                max_len,
                device=gene_tokens['gene_ids'].device,
                img_first_flag=self.img_first_flag,
                num_cell_in_tokens=self.num_query_tokens,
                num_cell_out_tokens=self.num_query_tokens,
                fortest=True,
            )
            for caption_ids in caption_ids_list
        ]

        text_tokens = {
            key: torch.stack([x[key] for x in tokenized_texts]) for key in tokenized_texts[0].keys()
        }
        text_tokens = BatchEncoding(data=text_tokens, tensor_type='pt')

        return text_tokens, gene_tokens, texts

class Stage2DM(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        vocab=None,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.cell_max_len = args.cell_max_len
        self.vocab = vocab
        self.pretrain_dataset = Stage1Dataset(root, vocab, debug_ratio=10, from_padding=True, test_split_num=0)
        # self.train_dataset = Stage1Dataset(root, vocab, debug_ratio=-1, from_padding=True)
        self.train_dataset = H5Dataset("path/to/data/zeroshot_cls/immune_tissue/immune_tissue.h5ad",
                                      text_max_len=text_max_len,
                                      vocab=vocab,
                                      bert_model_name="pubmedbert",
                                      ratio=0.9,
                                      equal_num=10000)
        self.val_dataset = Stage1Dataset(root, vocab, debug_ratio=20000, from_padding=True)
        # self.val_dataset.dataset = self.pretrain_dataset.test_dataset
        # print(f"{len(self.val_dataset)=}")
        # self.test_dataset = Stage1Dataset(root, vocab, debug_ratio=20000, from_padding=True)
        self.test_dataset = H5Dataset("path/to/data/zeroshot_cls/immune_tissue/immune_tissue.h5ad",
                                      text_max_len=text_max_len,
                                      vocab=vocab,
                                      bert_model_name="pubmedbert",
                                      ratio = -0.9,
                                      debug_ratio=0, equal_num=50)
        print(f"{len(self.train_dataset)=}")
        print(f"{len(self.test_dataset)=}")
        self.init_tokenizer(tokenizer)
        self.num_query_tokens = args.num_query_token
        self.img_first_flag = args.img_first_flag

    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        # self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, 
                                         self.text_max_len, 
                                         self.cell_max_len, 
                                         self.vocab, 
                                         self.num_query_tokens,
                                         self.img_first_flag),
            )
        elif self.mode in {'ft', 'train-test', 'eval'}:
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, 
                                         self.text_max_len, 
                                         self.cell_max_len, 
                                         self.vocab, 
                                         self.num_query_tokens,
                                         self.img_first_flag),
            )
        else:
            raise NotImplementedError
        return loader

    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, 
                                         self.text_max_len, 
                                         self.cell_max_len, 
                                         self.vocab, 
                                         self.num_query_tokens,
                                         self.img_first_flag),
            )

        test_loader = DataLoader(
            self.val_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TestCollater(self.tokenizer, 
                                         self.text_max_len, 
                                         self.cell_max_len, 
                                         self.vocab, 
                                         self.num_query_tokens,
                                         self.img_first_flag),
            )

        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TestCollater(self.tokenizer, 
                                         self.text_max_len, 
                                         self.cell_max_len, 
                                         self.vocab, 
                                         self.num_query_tokens,
                                         self.img_first_flag),
            )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--cell_max_len', type=int, default=2048, help='geneformer supports 4096')
        parser.add_argument('--img_first_flag', type=bool, default=None, help='img first')
        parser.add_argument('--prompt', type=str, default='')
        return parent_parser
    