import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.blip2_stage1 import Blip2Stage1
from data_provider.stage1_dm import Stage1DM
from utils import add_train_args
from model.cell_encoder import CellEncoder

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for A5000 gpus
torch.set_float32_matmul_precision('high') # can be medium (bfloat16), high (tensorfloat32), highest (float32)
from datasets import disable_caching
disable_caching()

def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, device=args.devices)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    dm = Stage1DM(
        args.root,
        args.zeroshot_cls_datafiles,
        args.zeroshot_retrieval_datafiles,
        model.blip2qformer.cell_encoder.vocab,
        args.num_workers,
        args.batch_size,
        args.match_batch_size,
        args.text_max_len,
        args.cell_max_len,
        args.bert_name,
        args
    )
    dm.train_dataset.tokenizer = model.blip2qformer.tokenizer
    dm.val_dataset.tokenizer = model.blip2qformer.tokenizer
    model.zeroshot_cls_loaders = dm.zeroshot_cls_loaders
    model.zeroshot_retrieval_loaders = dm.zeroshot_retrieval_loaders
    model.stage1_mode = args.mode

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1))
    callbacks.append(plc.TQDMProgressBar(refresh_rate=args.tqdm_interval))
    
    # find_unused_parameters = (not args.gtm) or (not args.lm)
    find_unused_parameters = True
    if len(args.devices.split(',')) > 1:
        strategy = strategies.DDPSpawnStrategy(find_unused_parameters=find_unused_parameters)
    else:
        strategy = None
        args.devices = eval(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger,
                                        #  limit_train_batches=100,
                                         )
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'continue train':
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49 ## avoid 
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_train_args(parser)

    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=False)
    parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=False)
    parser.add_argument('--tqdm_interval', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='path/to')
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM.add_model_specific_args(parser)
    parser = CellEncoder.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)
