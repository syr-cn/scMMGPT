import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from data_provider.stage2_dm import Stage2DM
# from data_provider.iupac_dm import IupacDM
from data_provider.stage2_chebi_dm import Stage2CheBIDM
from model.blip2_stage2 import Blip2Stage2
from model.cell_encoder import CellEncoder

# torch.set_default_dtype(torch.float16)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

class MyDDPSpawnStrategy(strategies.DDPSpawnStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path, args.opt_model)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)
    
    

    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.opt_model.find('galactica') >= 0 or args.opt_model.find('t5') >= 0:
        tokenizer = model.blip2lm.opt_tokenizer
    elif args.opt_model.lower().find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
        tokenizer = model.blip2lm.llm_tokenizer
    else:
        raise NotImplementedError
    # data
    dm = Stage2DM(args.mode, 
                    args.num_workers, 
                    args.batch_size, 
                    args.root, 
                    args.text_max_len, 
                    model.blip2lm.cell_encoder.vocab,
                    tokenizer, 
                    args)
    
    callbacks = []
    ## fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs,  
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    callbacks.append(plc.TQDMProgressBar(refresh_rate=args.tqdm_interval))

    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'fsdp':
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPSpawnStrategy(find_unused_parameters=True)
    else:
        strategy = None
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger,
                                        #  val_check_interval=args.val_check_interval,
                                        #  limit_train_batches=100,
                                         )
    # get_datasets(dm)
    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    if args.mode in {'train-test'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        # trainer.validate(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--iupac_prediction', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = Stage2DM.add_model_specific_args(parser)
    parser = CellEncoder.add_model_specific_args(parser)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--tqdm_interval', type=int, default=10)
    parser.add_argument('--valrank', action='store_true', default=False)
    parser.add_argument('--use_sentence', action='store_true', default=False)
    # parser.add_argument('--val_check_interval', type=float, default=1.0)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())
