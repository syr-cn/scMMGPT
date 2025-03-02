from pytorch_lightning import LightningDataModule
from data_provider.pretrain_dataset import Stage1Dataset
from data_provider.retrieval_dataset import ClsDataset, ClsCollater
from torch.utils.data import DataLoader
import os
import time
from scgpt.tokenizer import GeneVocab
from datasets import Dataset, load_dataset
from pathlib import Path
from tqdm import trange,tqdm
vocab = GeneVocab.from_file('scgpt_pretrained/human/vocab.json')

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, help='tissue of data')
    args = parser.parse_args()
    
    print(f"padding: {args.tissue}")

    data_path = f'./corpus_data/{args.tissue}'
    output_path = f"./cellxgene_hvalue/{args.tissue}"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    _dataset = Stage1Dataset(Path(data_path), vocab, debug_ratio=-1)

    _dataset.save_padding_dataset_hv(2048, output_path)

    print(f">>> padding end: {args.tissue}")

if __name__ == '__main__':
    main()
