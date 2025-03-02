import torch
import numpy as np
import os
import torch.utils
from torch.utils.data import Dataset, random_split
from datasets import load_dataset, concatenate_datasets, load_from_disk
from datasets import Dataset as HDataset
from pathlib import Path
import pyarrow.parquet as pq
import torch.utils.data
from transformers import BertTokenizer
import random
import json
import scanpy as sc
from .data_utils import process_adata_dir, CellTextHelper

def load_all_counts(data_source, num_proc=None):
    parquet_files = [str(f) for f in Path(data_source).glob("*.parquet")]
    parquet_files = [file for file in parquet_files if pq.read_metadata(file).num_columns > 0]
    if parquet_files == []:
        return None
    cache_dir = Path(data_source).parent / "cache"
    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=str(cache_dir),
        num_proc=num_proc,
    )
    return dataset


class Stage1Dataset(Dataset):
    def __init__(self, root, vocab, debug_ratio=0, from_padding=False, test_split_num=0):
        super(Stage1Dataset, self).__init__()
        self.root = root
        self.vocab = vocab
        self.test_split_num = test_split_num
        if from_padding:
            self.load_from_padding(root=root)
        else:
            self.load_from_root(root=root)
        self.cell_text_helper = CellTextHelper()
        

        # self.type2text = cell_to_text
        # self.type2index = {k.lower(): i for i, k in enumerate(self.type2text.keys())}
        # self.text_max_len = 128
        # self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        # for debugging 
        # self.dataset = self.dataset.shuffle(seed=42)
        if debug_ratio > 0:
            # print(Path(root).as_posix())
            sampled_indices = random.sample(range(len(self.dataset)), len(self.dataset)//debug_ratio)
            self.dataset = self.dataset.select(sampled_indices)

    def load_from_root(self, root):
        raw_path = Path(root) / 'raw/'
        if raw_path.exists():
            # single source of data
            processed_path = Path(root) / 'databanks/'
            if not processed_path.exists():
                process_adata_dir(raw_path, processed_path, vocab=self.vocab)
            self.dataset = load_all_counts(processed_path / 'all_counts')
        elif (Path(root) / 'all_counts').exists():
            self.dataset = load_all_counts(root / 'all_counts')
        else:
            # multiple sources of data, preprocessed already
            subdirs = [d for d in Path(root).iterdir() if d.is_dir() and (d/'all_counts').exists()]
            datasets = [load_all_counts(d / 'all_counts') for d in subdirs]
            datasets = [data for data in datasets if data is not None]
            self.dataset = concatenate_datasets(datasets)

    def load_from_padding(self, root):
        if self.test_split_num <= 0:
            self.dataset = self.load_from_padding_(root)
            return
        
        random.seed(42)
        subdirs = [d for d in Path(root).iterdir() if d.is_dir()]
        choose_datasets = []
        rest_datasets = []
        for d in subdirs:
            sub_dataset = self.load_from_padding_(d)
            sampled_indices = random.sample(range(len(sub_dataset)), self.test_split_num)
            rest_indices = set(range(len(sub_dataset))) - set(sampled_indices)
            rest_indices = list(rest_indices)

            choose = sub_dataset.select(sampled_indices)
            rest = sub_dataset.select(rest_indices)
            choose_datasets.append(choose)
            rest_datasets.append(rest)

        self.test_dataset = concatenate_datasets(choose_datasets)
        self.dataset = concatenate_datasets(rest_datasets)
        print(f"test dataset created! {len(self.test_dataset)=}")

    def load_from_padding_(self, root):
        info_path = Path(root) / 'dataset_info.json'
        if info_path.exists():
            return load_from_disk(root)
        else:
            subdirs = [d for d in Path(root).iterdir() if d.is_dir()]
            if subdirs == []:
                return None
            datasets = [self.load_from_padding_(d) for d in subdirs]
            datasets = [data for data in datasets if data is not None]
            if datasets == []:
                return None
            return concatenate_datasets(datasets)

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset[index]
        gene_ids = torch.tensor([self.vocab["<cls>"]] + data_item['genes'], dtype=torch.long)
        values = torch.tensor([-2] + data_item['expressions'], dtype=torch.float)
        metadata_keys = [k for k in data_item.keys() if k not in ['genes', 'expressions', 'id']]
        metadata = {k: data_item[k] for k in metadata_keys}
        # metadata = Stage1Dataset.update_cell_types(metadata)
        str_label = metadata['cell_type']
        # label = torch.tensor(self.type2index[str_label.lower()], dtype=torch.long)

        gene_inputs = {
            'id': data_item['id'],
            'gene_ids': gene_ids,
            'values': values,
            'metadata': metadata,
            "cell_type": str_label
            # 'label': label
        }
        text = self.cell_text_helper.cellxgene_metadata_to_text(metadata)
        return gene_inputs, text


    def get_text_list(self):
        text_list = [ f'This cell belongs to {k}: {self.type2text[k]}'
            for k in self.type2text.keys()]
        # print("\n\n@@@\n\n@@@\n\n@@@\n\n")
        # with open("temp_text_list.txt", "w") as ff:
        #     for ii in text_list:
        #         ff.write(ii)
        text_tokens, attention_masks = self.tokenizer_text(text_list)
        return text_tokens, attention_masks


    def save_padding_dataset_hv(self, padding_num, name, batch_size=200000):
        """
        保存 data_items, 但 'genes' 和 'expressions' 部分的长度被设为 padding_num-1
        """
        from tqdm import tqdm
        def process_data_batch(start_idx, end_idx,):
            padded_batch = []

            for idx in range(start_idx, end_idx):
                data_item = self.dataset[idx]
                if padding_num - 1 < len(data_item['genes']):
                    # # 随机选择 padding_num-1 个索引
                    # selected_indices = np.random.choice(len(data_item['genes']), padding_num-1, replace=False)

                    # # 获取选定索引的基因和表达
                    # padded_genes = [data_item['genes'][i] for i in selected_indices]
                    # padded_expressions = [data_item['expressions'][i] for i in selected_indices]
                    # idx = np.argsort(data_item['expressions'])[-(padding_num-1):]

                    # padded_genes = [data_item['genes'][i] for i in idx]
                    # padded_expressions = [data_item['expressions'][i] for i in idx]

                    # Assuming data_item is a dictionary with 'genes' and 'expressions' as keys
                    expressions = np.array(data_item['expressions'])
                    genes = np.array(data_item['genes'])

                    # Get the indices of the top (padding_num - 1) elements
                    idx = np.argpartition(expressions, -padding_num+1)[-padding_num+1:]

                    # Use the indices to select the elements from genes and expressions
                    padded_genes = genes[idx]
                    padded_expressions = expressions[idx]

                else:
                    padded_genes = data_item['genes']
                    padded_expressions = data_item['expressions']
                
                # 将基因和表达添加到字典中
                padded_item = {
                    'id': data_item['id'],
                    'genes': padded_genes,
                    'expressions': padded_expressions
                }

                # 添加其他 metadata 信息
                metadata_keys = [k for k in data_item.keys() if k not in ['genes', 'expressions', 'id']]
                metadata = {k: data_item[k] for k in metadata_keys}
                padded_item.update(metadata)

                padded_batch.append(padded_item)
            return padded_batch

        # padded_dataset = []
        total_batches = (len(self.dataset) + batch_size - 1) // batch_size  # 计算总批次数

        # 分批处理数据集
        for i in tqdm(range(total_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.dataset))
            print(f"processing [{start_idx}, {end_idx}]")
            # data_batch = self.dataset[start_idx:end_idx]
            padded_batch = process_data_batch(start_idx, end_idx)
            # padded_dataset.extend(padded_batch)

            # 每处理一个批次，保存到磁盘，减少内存占用
            temp_dataset = self.dataset.__class__.from_list(padded_batch)
            temp_dataset.save_to_disk(Path(name) / f"part{i}")

        # # 组合并保存完整的数据集
        # padded_dataset = self.dataset.__class__.from_list(padded_dataset)
        # padded_dataset.save_to_disk(Path(name))

    
class Stage1Dataset_tabula(Dataset):
    def __init__(self, root, vocab, debug_ratio=0, from_disk=False, sampled_indices=None, training_ratio=-1):
        super(Stage1Dataset_tabula, self).__init__()
        self.root = root
        self.vocab = vocab
        if from_disk:
            self.load_from_disk(root=root)
        else:
            self.load_from_root(root=root)

        self.return_text = training_ratio > 0
        self.cell_text_helper = CellTextHelper()

        if 'blood' in Path(root).as_posix(): 
            with open("path/to/tabula_blood_type.json", "r") as ff:
                self.type2text = json.load(ff)
        else:
            with open("path/to/tabula_all_type.json", "r") as ff:
                self.type2text = json.load(ff)       
        
        self.type2index = {k.lower(): i for i, k in enumerate(self.type2text.keys())}
        self.text_max_len = 128
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
        # self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        
        # for debugging 
        # self.dataset = self.dataset.shuffle(seed=42)
        if sampled_indices is not None:
            self.dataset = self.dataset.select(sampled_indices)
        if debug_ratio > 0:
            sampled_indices = random.sample(range(len(self.dataset)), len(self.dataset)//debug_ratio)
            self.dataset = self.dataset.select(sampled_indices)

        if training_ratio > 0:
            # with open('data/tabula/30_percent_types.json') as f:
            #     subset_types = json.load(f)
            # subset_types = [self.type2index[type_.lower()] for type_ in subset_types]
            subset_types = list(self.type2index.values())
            self.subset_types = random.sample(subset_types, int(len(self.type2index) * training_ratio))
            subset_types_log_file = 'data/tabula/random_subset_types.json'
            with open(subset_types_log_file, 'w') as f:
                json.dump(self.subset_types, f)
            # subset_types = subset_types[:int(len(self.type2text) * training_ratio)]
            print(f'Loaded {len(self.subset_types)} types for the training set.')

    def load_from_root(self, root):
        raw_path = Path(root) / 'raw/'
        if raw_path.exists():
            # single source of data
            processed_path = Path(root) / 'databanks/'
            if not processed_path.exists():
                process_adata_dir(raw_path, processed_path, vocab=self.vocab)
            self.dataset = load_all_counts(processed_path / 'all_counts')
        elif (Path(root) / 'all_counts').exists():
            self.dataset = load_all_counts(root / 'all_counts')
        else:
            # multiple sources of data, preprocessed already
            subdirs = [d for d in Path(root).iterdir() if d.is_dir() and (d/'all_counts').exists()]
            datasets = [load_all_counts(d / 'all_counts') for d in subdirs]
            datasets = [data for data in datasets if data is not None]
            self.dataset = concatenate_datasets(datasets)

    def load_from_disk(self, root):
        self.dataset = load_from_disk(root)

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if hasattr(self, 'subset_types'):
            str_label = self.dataset[index]['cell_type']
            label = self.type2index[str_label.lower()]
            if label not in self.subset_types:
                next_index = random.randint(0, len(self.dataset)-1)
                return self.__getitem__(next_index)
        data_item = self.dataset[index]
        gene_ids = torch.tensor([self.vocab["<cls>"]] + data_item['genes'], dtype=torch.long)
        values = torch.tensor([-2] + data_item['expressions'], dtype=torch.float)
        metadata_keys = [k for k in data_item.keys() if k not in ['genes', 'expressions', 'id']]
        metadata = {k: data_item[k] for k in metadata_keys}
        str_label = metadata['cell_type']
        label = torch.tensor(self.type2index[str_label.lower()], dtype=torch.long)

        gene_inputs = {
            # 'id': data_item['id'],
            'gene_ids': gene_ids,
            'values': values,
            # 'metadata': metadata,
            'label': label
        }
        if self.return_text:
            gene_inputs['id'] = data_item['id']
            gene_inputs['metadata'] = metadata
            text = self.cell_text_helper.cellxgene_metadata_to_text(metadata)
            return gene_inputs, text
        else:
            return gene_inputs

    def get_text_list(self):
        text_list = [ f'This cell belongs to {k}: {self.type2text[k]}'
            for k in self.type2text.keys()]
        # print("\n\n@@@\n\n@@@\n\n@@@\n\n")
        # with open("temp_text_list.txt", "w") as ff:
        #     for ii in text_list:
        #         ff.write(ii)
        text_tokens, attention_masks = self.tokenizer_text(text_list)
        return text_tokens, attention_masks

    
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