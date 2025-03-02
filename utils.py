import json
from multiprocessing import cpu_count
from pytorch_lightning import Trainer

def add_train_args(parent_parser):
    parent_parser = Trainer.add_argparse_args(parent_parser)
    parent_parser.set_defaults(
        accelerator='gpu',
        devices='0,1,2,3',
        precision='bf16',
        max_epochs=50,
        check_val_every_n_epoch=1
    )

    parser = parent_parser.add_argument_group("Trainer")
    parser.add_argument('--filename', default='none', type=str)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--root', type=str, default='data/PubChemDataset/PubChem-320k')
    parser.add_argument('--text_max_len', type=int, default=128)
    parser.add_argument('--cell_max_len', type=int, default=2048, help='geneformer supports 4096')
    return parent_parser


def save_json(data, filename):
    print(f'Writing {len(data)} data to {filename}...')
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f'Reading {len(data)} data from {filename}...')
    return data