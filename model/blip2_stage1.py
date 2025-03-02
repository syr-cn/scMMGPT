import torch
from model.blip2qformer import Blip2Qformer
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from tqdm import tqdm
from model.help_funcs import AttrDict
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
import os
import json

from pathlib import Path

class Blip2Stage1(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        
        self.args = args
        self.rerank_cand_num = args.rerank_cand_num
        self.temperature = args.temperature
        self.blip2qformer = Blip2Qformer(args.gtm, args.lm, args.bert_name, args.temperature, args.tune_gene_encoder, args.num_query_token, args.cross_attention_freq, args.projection_dim, args)
    
        self.save_hyperparameters(args)
        

    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = batch[0]['input_ids'].size(0)
        blip2_loss = self.blip2qformer(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss_gtc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_gtm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True)
        self.log("val_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)
        
        if (self.trainer.global_step + 1) % self.args.retrieval_eval_step == 0:
            self.do_validation()

        return blip2_loss.loss
    
    @torch.no_grad()
    def do_validation(self) -> None:
        for p, zeroshot_cls_loader in self.zeroshot_cls_loaders.items():
            # file_path = zeroshot_cls_loader.dataset.h5ad_path
            # filename = file_path.split('/')[-1].split('.')[0]
            file_path = "debug"
            filename = "debug"
            alpha_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            print(f'Evaluating {file_path}', flush=True)

            c2t_acc, c2t_f1, c2t_rec, c2t_rerank_acc_list, c2t_rerank_f1_list, c2t_rerank_rec_list = \
                eval_zero_shot_match_inbatch_with_rerank(self.blip2qformer, zeroshot_cls_loader, self.device, alpha_list)
            
            for alpha in alpha_list:
                print(f'Zero-shot Classification Result of {file_path}, alpha={alpha}', flush=True)
                print(f'c2t_acc: {c2t_acc}, c2t_rerank_acc: {c2t_rerank_acc_list[alpha]}', flush=True)
                print(f'c2t_f1: {c2t_f1}, c2t_rerank_f1: {c2t_rerank_f1_list[alpha]}', flush=True)
                for idx in range(10):
                    print(f'c2t_rec_{idx}: {c2t_rec[idx]},\tc2t_rerank_rec_{idx}: {c2t_rerank_rec_list[alpha][idx]}', flush=True)
                
                self.log(f'{filename}_c2t_acc', c2t_acc, sync_dist=False)
                self.log(f'{filename}_c2t_f1', c2t_f1, sync_dist=False)
                self.log(f'{filename}_c2t_rerank_acc', c2t_rerank_acc_list[alpha], sync_dist=False)
                self.log(f'{filename}_c2t_rerank_f1', c2t_rerank_f1_list[alpha], sync_dist=False)
            
    @torch.no_grad()
    def do_validation_multiGPU(self, self_rank) -> None:
        for path, zeroshot_cls_loader in self.zeroshot_cls_loaders.items():
            print(f'Evaluating on {path}', flush=True)
            filename = Path(path).name
            alpha_list = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

            c2t_acc, c2t_f1, c2t_rec, c2t_rerank_acc_list, c2t_rerank_f1_list, c2t_rerank_rec_list = \
                eval_zero_shot_match_inbatch_with_rerank_multiGPU(self.blip2qformer, zeroshot_cls_loader, self.device, alpha_list, self.temperature, self_rank)
            if self_rank == 0:
                for alpha in alpha_list:    
                    print(f'Zero-shot Classification Result of {path}, alpha={alpha}', flush=True)
                    print(f'c2t_acc: {c2t_acc}, c2t_rerank_acc: {c2t_rerank_acc_list[alpha]}', flush=True)
                    print(f'c2t_f1: {c2t_f1}, c2t_rerank_f1: {c2t_rerank_f1_list[alpha]}', flush=True)
                    for idx in range(10):
                        print(f'c2t_rec_{idx}: {c2t_rec[idx]},\tc2t_rerank_rec_{idx}: {c2t_rerank_rec_list[alpha][idx]}', flush=True)
                    
                    self.log(f'{filename}_c2t_acc', c2t_acc, sync_dist=False)
                    self.log(f'{filename}_c2t_f1', c2t_f1, sync_dist=False)
                    self.log(f'{filename}_c2t_rerank_acc', c2t_rerank_acc_list[alpha], sync_dist=False)
                    self.log(f'{filename}_c2t_rerank_f1', c2t_rerank_f1_list[alpha], sync_dist=False)

    def validation_epoch_end(self, outputs) -> None:
        if (self.current_epoch + 1) % self.args.retrieval_eval_epoch != 0:
            return
        
        if self.trainer.global_step == 0 and (self.stage1_mode == 'train' or self.stage1_mode == 'continue train'):
            return

        self.do_validation_multiGPU(self.trainer.global_rank)

        # if self.trainer.global_rank == 0:
        #     self.do_validation()

    
    def validation_epoch_end_molca(self, outputs) -> None:
        if (self.current_epoch + 1) % self.args.retrieval_eval_epoch != 0:
            return
        if self.trainer.global_rank == 0:
            ## for validation set
            g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, \
            g2t_rerank_acc, t2g_rerank_acc, g2t_rerank_rec20, t2g_rerank_rec20,\
            graph_rep_total, text_rep_total, _, _, _, _ = \
                eval_retrieval_inbatch_with_rerank(self.blip2qformer, self.val_match_loader, self.device)
            self.log("val_inbatch_g2t_acc", g2t_acc, sync_dist=False)
            self.log("val_inbatch_t2g_acc", t2g_acc, sync_dist=False)
            self.log("val_inbatch_g2t_rec20", g2t_rec20, sync_dist=False)
            self.log("val_inbatch_t2g_rec20", t2g_rec20, sync_dist=False)

            self.log("rerank_val_inbatch_g2t_acc", g2t_rerank_acc, sync_dist=False)
            self.log("rerank_val_inbatch_t2g_acc", t2g_rerank_acc, sync_dist=False)
            self.log("rerank_val_inbatch_g2t_rec20", g2t_rerank_rec20, sync_dist=False)
            self.log("rerank_val_inbatch_t2g_rec20", t2g_rerank_rec20, sync_dist=False)
            
            g2t_acc, g2t_rec20, t2g_acc, t2g_rec20, _ = \
                eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device)
            self.log("val_fullset_g2t_acc", g2t_acc, sync_dist=False)
            self.log("val_fullset_t2g_acc", t2g_acc, sync_dist=False)
            self.log("val_fullset_g2t_rec20", g2t_rec20, sync_dist=False)
            self.log("val_fullset_t2g_rec20", t2g_rec20, sync_dist=False)

            ## for test set
            g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, \
            g2t_rerank_acc, t2g_rerank_acc, g2t_rerank_rec20, t2g_rerank_rec20, \
            graph_rep_total, text_rep_total, graph_feat_total, graph_mask_total, text_total, text_mask_total = \
                eval_retrieval_inbatch_with_rerank(self.blip2qformer, self.test_match_loader, self.device)
            self.log("rerank_test_inbatch_g2t_acc", g2t_rerank_acc, sync_dist=False)
            self.log("rerank_test_inbatch_t2g_acc", t2g_rerank_acc, sync_dist=False)
            self.log("rerank_test_inbatch_g2t_rec20", g2t_rerank_rec20, sync_dist=False)
            self.log("rerank_test_inbatch_t2g_rec20", t2g_rerank_rec20, sync_dist=False)

            self.log("test_inbatch_g2t_acc", g2t_acc, sync_dist=False)
            self.log("test_inbatch_t2g_acc", t2g_acc, sync_dist=False)
            self.log("test_inbatch_g2t_rec20", g2t_rec20, sync_dist=False)
            self.log("test_inbatch_t2g_rec20", t2g_rec20, sync_dist=False)
            
            g2t_acc, g2t_rec20, t2g_acc, t2g_rec20, sim_g2t = \
                eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device)
            self.log("test_fullset_g2t_acc", g2t_acc, sync_dist=False)
            self.log("test_fullset_t2g_acc", t2g_acc, sync_dist=False)
            self.log("test_fullset_g2t_rec20", g2t_rec20, sync_dist=False)
            self.log("test_fullset_t2g_rec20", t2g_rec20, sync_dist=False)

            g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = \
                eval_retrieval_fullset_for_rerank(self.blip2qformer, sim_g2t, graph_feat_total, graph_mask_total, text_total, text_mask_total, self.rerank_cand_num, self.device)
            self.log("rerank_test_fullset_g2t_acc", g2t_acc, sync_dist=False)
            self.log("rerank_test_fullset_t2g_acc", t2g_acc, sync_dist=False)
            self.log("rerank_test_fullset_g2t_rec20", g2t_rec20, sync_dist=False)
            self.log("rerank_test_fullset_t2g_rec20", t2g_rec20, sync_dist=False)
            del graph_rep_total, text_rep_total

    def training_step(self, batch, batch_idx):
        # if self.trainer.global_step < self.args.warmup_steps:
        #     warmup_lr_schedule(self.trainer.optimizers[0], self.trainer.global_step, self.args.warmup_steps, self.args.warmup_lr, self.args.init_lr)
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[-1]['gene_ids'].size(0)
        blip2_loss = self.blip2qformer(batch)
        ###============== Overall Loss ===================###
        self.log("train_loss_gtc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("train_loss_gtm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("train_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("train_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True, prog_bar=True)


        return blip2_loss.loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.args.validation_every_step > 0 and (self.trainer.global_step + 1) % self.args.validation_every_step == 0: 
            self.do_validation_multiGPU(self.trainer.global_rank)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')

        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        # evaluation
        parser.add_argument('--rerank_cand_num', type=int, default=128)
        
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gene_encoder', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--projection_dim', type=int, default=256)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        parser.add_argument('--batch_size_contrast', type=int, default=-1)
        parser.add_argument('--qformer_contrast_func', type=str, default="contrast", help="[contrast, contrast_batch, contrast_batch_sim]")
        parser.add_argument('--validation_every_step', type=int, default=0, help="validation_every_step")
        parser.add_argument('--use_flash_attn', action='store_true', default=False)
        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--retrieval_eval_epoch', type=int, default=10)
        parser.add_argument('--retrieval_eval_step', type=int, default=-1)
        return parent_parser


def pad_and_concat(tensor_list):
    '''
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    '''
    device = tensor_list[0].device
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.zeros((sum_dim0, max_dim1, tensor_list[0].shape[-1]), device=device)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.zeros((sum_dim0, max_dim1), device=device)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()

@torch.no_grad()
def eval_retrieval_inbatch(model, dataloader, device=None):
    assert isinstance(model, Blip2Qformer)
    model.eval()
    g2t_acc = 0
    t2g_acc = 0
    g2t_rec20 = 0
    t2g_rec20 = 0
    allcnt = 0
    
    graph_rep_total = []  
    text_rep_total = []
    
    graph_feat_total = [] 
    graph_mask_total = []
    
    text_total = []
    text_mask_total = []
    
    for batch in tqdm(dataloader):
        aug, text, text_mask = batch
        text_total.append(text)
        text_mask_total.append(text_mask)

        aug = aug.to(device)
        text = text.to(device)
        text_mask = text_mask.to(device)
        graph_rep, graph_feat, graph_mask = model.graph_forward(aug) # shape = [B, num_qs, D]
        text_rep = model.text_forward(text, text_mask) # shape = [B, D]

        sim_q2t = (graph_rep.unsqueeze(1) @ text_rep.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        B = sim_g2t.shape[0]
        sorted_ids = sim_g2t.argsort(descending=True).cpu()
        g2t_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        sorted_ids = sim_g2t.T.argsort(descending=True).cpu()
        t2g_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        # argm1 = torch.argmax(sim_g2t, axis=1)
        # argm2 = torch.argmax(sim_g2t.T, axis=1)

        g2t_acc += float((g2t_rank == 0).sum())
        t2g_acc += float((t2g_rank == 0).sum())
        g2t_rec20 += float((g2t_rank < 20).sum())
        t2g_rec20 += float((t2g_rank < 20).sum())
        
        allcnt += B

        graph_rep_total.append(graph_rep.cpu())
        text_rep_total.append(text_rep.cpu())
        graph_feat_total.append(graph_feat.cpu())
        graph_mask_total.append(graph_mask.cpu())

    graph_rep_total = torch.cat(graph_rep_total, dim=0)
    text_rep_total = torch.cat(text_rep_total, dim=0)
    graph_feat_total = pad_and_concat(graph_feat_total)
    graph_mask_total = pad_and_concat(graph_mask_total)
    text_total = torch.cat(text_total, dim=0)
    text_mask_total = torch.cat(text_mask_total, dim=0)

    g2t_acc = round(g2t_acc/allcnt * 100, 2)
    t2g_acc = round(t2g_acc/allcnt * 100, 2)
    g2t_rec20 = round(g2t_rec20 / allcnt * 100, 2)
    t2g_rec20 = round(t2g_rec20 / allcnt * 100, 2)
    return g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total, graph_feat_total, graph_mask_total, text_total, text_mask_total


@torch.no_grad()
def eval_retrieval_fullset(graph_rep, text_rep, device):    
    N = graph_rep.shape[0]
    B = 8
    text_rep = text_rep.to(device)
    sim_g2t = []
    for i in tqdm(range(0, N, B)):
        l_graph_rep = graph_rep[i:i+B].to(device)
        l_sim_q2t = (l_graph_rep.unsqueeze(1) @ text_rep.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [N, D, 1]; output shape = [B, N, num_qs]
        l_sim_g2t, _ = l_sim_q2t.max(-1) # shape = [B, N]
        sim_g2t.append(l_sim_g2t)
    sim_g2t = torch.cat(sim_g2t, dim=0).cpu() # shape = [N, N]
    
    rank_g2t = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_g2t[i:i+B].to(device), descending=True)
        rank_g2t.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_g2t = torch.cat(rank_g2t, dim=0)
    
    rank_t2g = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_g2t.T[i:i+B].to(device), descending=True)
        rank_t2g.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_t2g = torch.cat(rank_t2g, dim=0)
    
    g2t_acc = float((rank_g2t == 0).float().mean())
    g2t_rec20 = float((rank_g2t < 20).float().mean())
    t2g_acc = float((rank_t2g == 0).float().mean())
    t2g_rec20 = float((rank_t2g < 20).float().mean())
    g2t_acc = round(g2t_acc * 100, 2)
    g2t_rec20 = round(g2t_rec20 * 100, 2)
    t2g_acc = round(t2g_acc * 100, 2)
    t2g_rec20 = round(t2g_rec20 * 100, 2)
    return g2t_acc, g2t_rec20, t2g_acc, t2g_rec20, sim_g2t


@torch.no_grad()
def eval_retrieval_fullset_for_rerank(model, sim_g2t_total, graph_feat_total, graph_mask_total, text_total, text_mask_total, rerank_cand_num, device):
    N = sim_g2t_total.shape[0]
    B = 4    
    rcn = rerank_cand_num ## re-rank candidate numbers
    
    hit_g2t = []
    for i in tqdm(range(0, N, B), desc='re-ranking g2t'):
        sim = sim_g2t_total[i:i+B].to(device)
        rB = sim.shape[0] # real batch size
        topk_sim, topk_idx = sim.topk(k=rcn, dim=1) # shape = [B, rcn]
        topk_idx = topk_idx.cpu()
        graph_feat = graph_feat_total[i:i+B].to(device).repeat_interleave(rcn, 0) # shape = [B * rcn, num_qs, D]
        graph_mask = graph_mask_total[i:i+B].to(device).repeat_interleave(rcn, 0) # shape = [B * rcn, num_qs, D]
        text = text_total[topk_idx].flatten(0,1).to(device) # shape = [B * rcn, text_len]
        text_mask = text_mask_total[topk_idx].flatten(0,1).to(device) # shape = [B * rcn, text_len]
        gtm_sim = model.compute_gtm(graph_feat, graph_mask, text, text_mask).reshape(rB, rcn) ## fixme, using the linear clf's logits directly, without softmax
        sorted_ids = torch.argsort(topk_sim + gtm_sim, descending=True).cpu() # shape = [B, rcn]
        # sorted_ids = torch.argsort(gtm_sim, descending=True).cpu() # shape = [B, rcn]
        sorted_ids = torch.gather(topk_idx, 1, sorted_ids) # mapping to original ids
        hit_g2t.append((sorted_ids == torch.arange(i,i+rB).reshape(-1, 1)).int())
    
    hit_g2t = torch.cat(hit_g2t, dim=0) # shape = [N, rcn]
    # g2t_acc = float((hit_g2t[:, 0]).float().mean())
    # g2t_rec20 = float((hit_g2t[:, :20]).float().sum() / N)
    # print(g2t_acc, g2t_rec20)

    hit_t2g = []
    sim_t2g_total = sim_g2t_total.T
    for i in tqdm(range(0, N, B), desc='re-ranking t2g'):
        sim = sim_t2g_total[i:i+B].to(device)
        rB = sim.shape[0]
        topk_sim, topk_idx = sim.topk(k=rcn, dim=1)
        topk_idx = topk_idx.cpu()
        text = text_total[i:i+B].to(device).repeat_interleave(rcn, 0)
        text_mask = text_mask_total[i:i+B].to(device).repeat_interleave(rcn, 0)
        graph_feat = graph_feat_total[topk_idx].to(device).flatten(0,1)
        graph_mask = graph_mask_total[topk_idx].to(device).flatten(0,1)
        gtm_sim = model.compute_gtm(graph_feat, graph_mask, text, text_mask).reshape(rB, rcn)
        sorted_ids = torch.argsort(topk_sim + gtm_sim, descending=True).cpu() # shape = [B, rcn]
        sorted_ids = torch.gather(topk_idx, 1, sorted_ids)
        hit_t2g.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0]).reshape(-1, 1)).int())
    hit_t2g = torch.cat(hit_t2g, dim=0)
    
    g2t_acc = float((hit_g2t[:, 0]).float().mean())
    g2t_rec20 = float((hit_g2t[:, :20]).float().sum() / N)
    t2g_acc = float((hit_t2g[:, 0]).float().mean())
    t2g_rec20 = float((hit_t2g[:, :20]).float().sum() / N)
    g2t_acc = round(g2t_acc * 100, 2)
    g2t_rec20 = round(g2t_rec20 * 100, 2)
    t2g_acc = round(t2g_acc * 100, 2)
    t2g_rec20 = round(t2g_rec20 * 100, 2)
    return g2t_acc, g2t_rec20, t2g_acc, t2g_rec20


@torch.no_grad()
def eval_retrieval_inbatch_with_rerank(model, dataloader, device=None):
    '''
    include rerank
    '''
    assert isinstance(model, Blip2Qformer)
    model.eval()
    g2t_acc = 0
    t2g_acc = 0
    g2t_rec20 = 0
    t2g_rec20 = 0
    allcnt = 0
    
    g2t_rerank_acc = 0
    t2g_rerank_acc = 0
    g2t_rerank_rec20 = 0
    t2g_rerank_rec20 = 0

    graph_rep_total = []  
    text_rep_total = []
    
    graph_feat_total = [] 
    graph_mask_total = []
    
    text_total = []
    text_mask_total = []
    
    for batch in tqdm(dataloader):
        aug, text, text_mask = batch
        text_total.append(text)
        text_mask_total.append(text_mask)

        aug = aug.to(device)
        text = text.to(device)
        text_mask = text_mask.to(device)
        graph_rep, graph_feat, graph_mask = model.graph_forward(aug) # shape = [B, num_qs, D]
        text_rep = model.text_forward(text, text_mask) # shape = [B, D]

        sim_q2t = (graph_rep.unsqueeze(1) @ text_rep.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        B = sim_g2t.shape[0]
        sorted_ids = sim_g2t.argsort(descending=True).cpu()
        g2t_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        sorted_ids = sim_g2t.T.argsort(descending=True).cpu()
        t2g_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        
        g2t_acc += float((g2t_rank == 0).sum())
        t2g_acc += float((t2g_rank == 0).sum())
        g2t_rec20 += float((g2t_rank < 20).sum())
        t2g_rec20 += float((t2g_rank < 20).sum())

        allcnt += B

        graph_rep_total.append(graph_rep.cpu())
        text_rep_total.append(text_rep.cpu())
        graph_feat_total.append(graph_feat.cpu())
        graph_mask_total.append(graph_mask.cpu())

        ## reranking
        graph_feat = graph_feat.repeat_interleave(B, 0) # shape = [B * B, num_qs, D]
        graph_mask = graph_mask.repeat_interleave(B, 0) # shape = [B * B, num_qs, D]
        text = text.repeat(B, 1) # shape = [B * B, text_len]
        text_mask = text_mask.repeat(B, 1) # shape = [B * B, text_len]

        if False:
            gtm_sim = model.compute_gtm(graph_feat, graph_mask, text, text_mask).reshape(B, B)
        else:
            ## batched reranking
            batch_size = 64
            gtm_sim = []
            for i in range(0, graph_feat.shape[0], batch_size):
                gtm_sim_local = model.compute_gtm(graph_feat[i:i+batch_size], graph_mask[i:i+batch_size], text[i:i+batch_size], text_mask[i:i+batch_size])
                gtm_sim.append(gtm_sim_local)
            gtm_sim = torch.cat(gtm_sim, dim=0).reshape(B, B)

        rerank_sim = sim_g2t + gtm_sim

        ## g2t rerank
        sorted_ids = torch.argsort(rerank_sim, descending=True).cpu() # shape = [B, B]
        hit_g2t = (sorted_ids == torch.arange(B).reshape(-1, 1)).float()
        g2t_rerank_acc += float(hit_g2t[:, 0].sum())
        g2t_rerank_rec20 += float(hit_g2t[:, :20].sum())
        
        ## t2g rerank
        sorted_ids = torch.argsort(rerank_sim.T, descending=True).cpu() # shape = [B, B]
        hit_t2g = (sorted_ids == torch.arange(B).reshape(-1, 1)).float()
        t2g_rerank_acc += float(hit_t2g[:, 0].sum())
        t2g_rerank_rec20 += float(hit_t2g[:, :20].sum())

    graph_rep_total = torch.cat(graph_rep_total, dim=0)
    text_rep_total = torch.cat(text_rep_total, dim=0)
    graph_feat_total = pad_and_concat(graph_feat_total)
    graph_mask_total = pad_and_concat(graph_mask_total)
    text_total = torch.cat(text_total, dim=0)
    text_mask_total = torch.cat(text_mask_total, dim=0)

    g2t_acc = round(g2t_acc/allcnt * 100, 2)
    t2g_acc = round(t2g_acc/allcnt * 100, 2)
    g2t_rec20 = round(g2t_rec20 / allcnt * 100, 2)
    t2g_rec20 = round(t2g_rec20 / allcnt * 100, 2)

    g2t_rerank_acc = round(g2t_rerank_acc / allcnt * 100, 2)
    t2g_rerank_acc = round(t2g_rerank_acc / allcnt * 100, 2)
    g2t_rerank_rec20 = round(g2t_rerank_rec20 / allcnt * 100, 2)
    t2g_rerank_rec20 = round(t2g_rerank_rec20 / allcnt * 100, 2)
    return g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, \
        g2t_rerank_acc, t2g_rerank_acc, g2t_rerank_rec20, t2g_rerank_rec20, \
        graph_rep_total, text_rep_total, graph_feat_total, graph_mask_total, text_total, text_mask_total

@torch.no_grad()
def eval_zero_shot_match_inbatch_with_rerank(model, dataloader, device=None, alpha_list=[0,0.1,1]):
    '''
    include rerank
    '''
    REC_NUM = 10
    assert isinstance(model, Blip2Qformer)
    model.eval()

    c2t_acc = 0
    c2t_rec = [0 for _ in range(REC_NUM)]
    allcnt = 0
    
    c2t_rerank_acc_list = {alpha: 0 for alpha in alpha_list}
    c2t_rerank_f1_list = {alpha: 0 for alpha in alpha_list}
    c2t_rerank_rec_list = {alpha: [0 for _ in range(REC_NUM)] for alpha in alpha_list}
    
    text, text_mask = dataloader.dataset.get_text_list()
    text = text.to(device)
    text_mask = text_mask.to(device)
    text_feats = model.text_forward(text, text_mask) # shape = [NUM_CLS, D]
    NUM_CLS = text_feats.shape[0]
    all_labels = []
    all_preds = []
    all_preds_rerank_dict = {alpha: [] for alpha in alpha_list}

    for batch in tqdm(dataloader):
        gene_inputs, labels = batch
        for k in gene_inputs:
            gene_inputs[k] = gene_inputs[k].to(device)
        labels = labels.to(device) # [B], in range [0, NUM_CLS)

        gene_feats, cell_feats, gene_mask = model.cell_forward(gene_inputs) # cell_feats: [B, Nq, D]; gene_mask: [B, Nq]

        # Cell-Text Contrastive Similairty
        sim_q2t = (cell_feats.unsqueeze(1) @ text_feats.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_c2t, _ = sim_q2t.max(-1) # shape = [B, NUM_CLS]

        # Eval Before Rerank
        B = sim_c2t.shape[0]
        allcnt += B
        sorted_ids = torch.argsort(sim_c2t, descending=True) # shape = [B, NUM_CLS]
        hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
        c2t_acc += float(hit_c2t[:, 0].sum())
        for i in range(REC_NUM):
            c2t_rec[i] += float(hit_c2t[:, :i+1].sum())
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(sorted_ids[:, 0].cpu().tolist())

        # Cell-Text Matching Similairty
        gene_feats_repeat = gene_feats.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs, D]
        gene_mask_repeat = gene_mask.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs]
        text_repeat = text.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        text_mask_repeat = text_mask.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        ctm_sim = model.compute_gtm(gene_feats_repeat, gene_mask_repeat, text_repeat, text_mask_repeat).reshape(B, NUM_CLS)

        # Eval After Rerank
        for alpha in alpha_list:
            rerank_sim = alpha * sim_c2t + (1-alpha) * ctm_sim
            sorted_ids = torch.argsort(rerank_sim, descending=True) # shape = [B, NUM_CLS]
            hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
            c2t_rerank_acc_list[alpha] += float(hit_c2t[:, 0].sum())
            for i in range(REC_NUM):
                c2t_rerank_rec_list[alpha][i] += float(hit_c2t[:, :i+1].sum())
            all_preds_rerank_dict[alpha].extend(sorted_ids[:, 0].cpu().tolist()) ### #

    c2t_acc = round(c2t_acc / allcnt * 100, 2)
    for i in range(REC_NUM):
        c2t_rec[i] = round(c2t_rec[i] / allcnt * 100, 2)
    c2t_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    c2t_f1 = round(c2t_f1, 2)

    for alpha in alpha_list:
        c2t_rerank_acc_list[alpha] = round(c2t_rerank_acc_list[alpha] / allcnt * 100, 2)
        for i in range(REC_NUM):
            c2t_rerank_rec_list[alpha][i] = round(c2t_rerank_rec_list[alpha][i] / allcnt * 100, 2)
        c2t_rerank_f1_list[alpha] = f1_score(all_labels, all_preds_rerank_dict[alpha], average='macro') * 100
        c2t_rerank_f1_list[alpha] = round(c2t_rerank_f1_list[alpha], 2)
        cls_report = readout_zeroshot_report(all_labels, all_preds_rerank_dict[alpha])
        
        print(f"{alpha=}: ")
        print(cls_report)

    return c2t_acc, c2t_f1, c2t_rec, c2t_rerank_acc_list, c2t_rerank_f1_list, c2t_rerank_rec_list


@torch.no_grad()
def eval_zero_shot_match_inbatch_with_rerank_nouse(model, dataloader, device=None, alpha=0.5):
    '''
    include rerank
    '''
    REC_NUM = 10
    assert isinstance(model, Blip2Qformer)
    model.eval()
    c2t_acc = 0
    c2t_rec = [0 for _ in range(REC_NUM)]
    allcnt = 0
    
    c2t_rerank_acc = 0
    c2t_rerank_rec = [0 for _ in range(REC_NUM)]
    
    text, text_mask = dataloader.dataset.get_text_list()
    text = text.to(device)
    text_mask = text_mask.to(device)
    text_feats = model.text_forward(text, text_mask) # shape = [NUM_CLS, D]
    NUM_CLS = text_feats.shape[0]
    all_labels = []
    all_preds = []
    all_preds_rerank = []

    for batch in tqdm(dataloader):
        gene_inputs, labels = batch

        for k in gene_inputs:
            gene_inputs[k] = gene_inputs[k].to(device)
        labels = labels.to(device) # [B], in range [0, NUM_CLS)

        gene_feats, cell_feats, gene_mask = model.cell_forward(gene_inputs) # cell_feats: [B, Nq, D]; gene_mask: [B, Nq]

        # Cell-Text Contrastive Similairty
        sim_q2t = (cell_feats.unsqueeze(1) @ text_feats.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_c2t, _ = sim_q2t.max(-1) # shape = [B, NUM_CLS]

        # Eval Before Rerank
        B = sim_c2t.shape[0]
        allcnt += B
        sorted_ids = torch.argsort(sim_c2t, descending=True) # shape = [B, NUM_CLS]
        hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
        c2t_acc += float(hit_c2t[:, 0].sum())
        for i in range(REC_NUM):
            c2t_rec[i] += float(hit_c2t[:, :i+1].sum())
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(sorted_ids[:, 0].cpu().tolist())

        # Cell-Text Matching Similairty
        gene_feats_repeat = gene_feats.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs, D]
        gene_mask_repeat = gene_mask.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs]
        text_repeat = text.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        text_mask_repeat = text_mask.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        ctm_sim = model.compute_gtm(gene_feats_repeat, gene_mask_repeat, text_repeat, text_mask_repeat).reshape(B, NUM_CLS)

        # Eval After Rerank
        rerank_sim = alpha * sim_c2t + (1-alpha) * ctm_sim
        sorted_ids = torch.argsort(rerank_sim, descending=True) # shape = [B, NUM_CLS]
        hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
        c2t_rerank_acc += float(hit_c2t[:, 0].sum())
        for i in range(REC_NUM):
            c2t_rerank_rec[i] += float(hit_c2t[:, :i+1].sum())
        all_preds_rerank.extend(sorted_ids[:, 0].cpu().tolist()) ### #


    c2t_acc = round(c2t_acc / allcnt * 100, 2)
    for i in range(REC_NUM):
        c2t_rec[i] = round(c2t_rec[i] / allcnt * 100, 2)
    c2t_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    c2t_f1 = round(c2t_f1, 2)

    c2t_rerank_acc = round(c2t_rerank_acc / allcnt * 100, 2)
    for i in range(REC_NUM):
        c2t_rerank_rec[i] = round(c2t_rerank_rec[i] / allcnt * 100, 2)
    c2t_rerank_f1 = f1_score(all_labels, all_preds_rerank, average='macro') * 100
    c2t_rerank_f1 = round(c2t_rerank_f1, 2)

    return c2t_acc, c2t_f1, c2t_rec, c2t_rerank_acc, c2t_rerank_f1, c2t_rerank_rec

@torch.no_grad()
def eval_zero_shot_match_inbatch_with_rerank_tmp(model, dataloader, device=None, alpha_list=[0,0.05,1]):
    '''
    include rerank
    '''
    REC_NUM = 10
    assert isinstance(model, Blip2Qformer)
    model.eval()

    c2t_acc = 0
    c2t_rec = [0 for _ in range(REC_NUM)]
    allcnt = 0
    
    c2t_rerank_acc_list = {alpha: 0 for alpha in alpha_list}
    c2t_rerank_f1_list = {alpha: 0 for alpha in alpha_list}
    c2t_rerank_rec_list = {alpha: [0 for _ in range(REC_NUM)] for alpha in alpha_list}
    
    text, text_mask = dataloader.dataset.get_text_list()
    text = text.to(device)
    text_mask = text_mask.to(device)
    text_feats = model.text_forward(text, text_mask) # shape = [NUM_CLS, D]
    NUM_CLS = text_feats.shape[0]
    all_labels = []
    all_preds = []
    all_preds_rerank_dict = {alpha: [] for alpha in alpha_list}

    for batch in tqdm(dataloader):
        gene_inputs, labels = batch

        for k in gene_inputs:
            gene_inputs[k] = gene_inputs[k].to(device)
        labels = labels.to(device) # [B], in range [0, NUM_CLS)

        gene_feats, cell_feats, gene_mask = model.cell_forward(gene_inputs) # cell_feats: [B, Nq, D]; gene_mask: [B, Nq]

        # Cell-Text Contrastive Similairty
        sim_q2t = (cell_feats.unsqueeze(1) @ text_feats.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_c2t, _ = sim_q2t.max(-1) # shape = [B, NUM_CLS]

        # Eval Before Rerank
        B = sim_c2t.shape[0]
        allcnt += B
        sorted_ids = torch.argsort(sim_c2t, descending=True) # shape = [B, NUM_CLS]
        hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
        c2t_acc += float(hit_c2t[:, 0].sum())
        for i in range(REC_NUM):
            c2t_rec[i] += float(hit_c2t[:, :i+1].sum())
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(sorted_ids[:, 0].cpu().tolist())

        # Cell-Text Matching Similairty
        gene_feats_repeat = gene_feats.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs, D]
        gene_mask_repeat = gene_mask.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs]
        text_repeat = text.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        text_mask_repeat = text_mask.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        ctm_sim = model.compute_gtm(gene_feats_repeat, gene_mask_repeat, text_repeat, text_mask_repeat).reshape(B, NUM_CLS)

        # Eval After Rerank
        for alpha in alpha_list:
            rerank_sim = alpha * sim_c2t + (1-alpha) * ctm_sim
            sorted_ids = torch.argsort(rerank_sim, descending=True) # shape = [B, NUM_CLS]
            hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
            c2t_rerank_acc_list[alpha] += float(hit_c2t[:, 0].sum())
            for i in range(REC_NUM):
                c2t_rerank_rec_list[alpha][i] += float(hit_c2t[:, :i+1].sum())
            all_preds_rerank_dict[alpha].extend(sorted_ids[:, 0].cpu().tolist()) ### #

    c2t_acc = round(c2t_acc / allcnt * 100, 2)
    for i in range(REC_NUM):
        c2t_rec[i] = round(c2t_rec[i] / allcnt * 100, 2)
    c2t_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    c2t_f1 = round(c2t_f1, 2)

    for alpha in alpha_list:
        c2t_rerank_acc_list[alpha] = round(c2t_rerank_acc_list[alpha] / allcnt * 100, 2)
        for i in range(REC_NUM):
            c2t_rerank_rec_list[alpha][i] = round(c2t_rerank_rec_list[alpha][i] / allcnt * 100, 2)
        c2t_rerank_f1_list[alpha] = f1_score(all_labels, all_preds_rerank_dict[alpha], average='macro') * 100
        c2t_rerank_f1_list[alpha] = round(c2t_rerank_f1_list[alpha], 2)
        cls_report = readout_zeroshot_report(all_labels, all_preds_rerank_dict[alpha])
        print(f"{alpha=}: ")
        print(cls_report)

    return c2t_acc, c2t_f1, c2t_rec, c2t_rerank_acc_list, c2t_rerank_f1_list, c2t_rerank_rec_list

import torch.distributed as dist

@torch.no_grad()
def eval_zero_shot_match_inbatch_with_rerank_multiGPU(model, dataloader, device=None, alpha_list=[0,0.05,1], temperature=0.1, self_rank=1):
    '''
    include rerank
    '''
    REC_NUM = 10
    assert isinstance(model, Blip2Qformer)
    model.eval()

    c2t_acc = 0
    c2t_rec = [0 for _ in range(REC_NUM)]
    allcnt = 0
    
    c2t_rerank_acc_list = {alpha: 0 for alpha in alpha_list}
    c2t_rerank_f1_list = {alpha: 0 for alpha in alpha_list}
    c2t_rerank_rec_list = {alpha: [0 for _ in range(REC_NUM)] for alpha in alpha_list}
    
    text, text_mask = dataloader.dataset.get_text_list()
    text = text.to(device)
    text_mask = text_mask.to(device)
    text_feats = model.text_forward(text, text_mask) # shape = [NUM_CLS, D]
    NUM_CLS = text_feats.shape[0]

    local_c2t_acc = 0
    local_c2t_rec = [0 for _ in range(REC_NUM)]
    local_allcnt = 0
    local_all_labels = []
    local_all_preds = []
    local_all_preds_rerank_dict = {alpha: [] for alpha in alpha_list}
    local_c2t_rerank_acc_list = {alpha: 0 for alpha in alpha_list}
    local_c2t_rerank_rec_list = {alpha: [0 for _ in range(REC_NUM)] for alpha in alpha_list}

    world_size = dist.get_world_size()

    for batch_idx, batch in enumerate(tqdm(dataloader, f"{self_rank=}", mininterval=10)):
        if batch_idx % world_size != self_rank:
            continue

        gene_inputs, labels = batch

        for k in gene_inputs:
            gene_inputs[k] = gene_inputs[k].to(device)
        labels = labels.to(device) # [B], in range [0, NUM_CLS)

        gene_feats, cell_feats, gene_mask = model.cell_forward(gene_inputs) # cell_feats: [B, Nq, D]; gene_mask: [B, Nq]

        # Cell-Text Contrastive Similarity
        sim_q2t = (cell_feats.unsqueeze(1) @ text_feats.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_q2t = sim_q2t / temperature
        sim_c2t, _ = sim_q2t.max(-1) # shape = [B, NUM_CLS]

        # Eval Before Rerank
        B = sim_c2t.shape[0]
        local_allcnt += B
        sorted_ids = torch.argsort(sim_c2t, descending=True) # shape = [B, NUM_CLS]
        hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
        local_c2t_acc += float(hit_c2t[:, 0].sum())
        for i in range(REC_NUM):
            local_c2t_rec[i] += float(hit_c2t[:, :i+1].sum())
        local_all_labels.extend(labels.cpu().tolist())
        local_all_preds.extend(sorted_ids[:, 0].cpu().tolist())

        # Cell-Text Matching Similarity
        gene_feats_repeat = gene_feats.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs, D]
        gene_mask_repeat = gene_mask.repeat_interleave(NUM_CLS, 0) # shape = [B * NUM_CLS, num_qs]
        text_repeat = text.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        text_mask_repeat = text_mask.repeat(B, 1) # shape = [B * NUM_CLS, text_len]
        ctm_sim = model.compute_gtm(gene_feats_repeat, gene_mask_repeat, text_repeat, text_mask_repeat).reshape(B, NUM_CLS)

        # Eval After Rerank
        for alpha in alpha_list:
            rerank_sim = alpha * sim_c2t + (1-alpha) * ctm_sim
            sorted_ids = torch.argsort(rerank_sim, descending=True) # shape = [B, NUM_CLS]
            hit_c2t = (sorted_ids == labels.reshape(-1, 1)).int()
            local_c2t_rerank_acc_list[alpha] += float(hit_c2t[:, 0].sum())
            for i in range(REC_NUM):
                local_c2t_rerank_rec_list[alpha][i] += float(hit_c2t[:, :i+1].sum())
            local_all_preds_rerank_dict[alpha].extend(sorted_ids[:, 0].cpu().tolist()) ### #

    # Use all_gather to collect results from all processes
    all_local_labels = [None for _ in range(world_size)]
    all_local_preds = [None for _ in range(world_size)]
    all_c2t_acc = [None for _ in range(world_size)]
    all_c2t_rec = []
    for i in range(REC_NUM):
        all_c2t_rec.append([None for _ in range(world_size)])
    all_allcnt = [None for _ in range(world_size)]
    all_local_preds_rerank_dict = {alpha: [None for _ in range(world_size)] for alpha in alpha_list}
    all_c2t_rerank_acc_list = {alpha: [None for _ in range(world_size)] for alpha in alpha_list}
    all_c2t_rerank_rec_list = {alpha: [None for _ in range(world_size)] for alpha in alpha_list}

    # print(f"before all_gather: Current memory allocated on {device}: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB") 
    # print(f"before all_gather: Max memory allocated on {device}: {torch.cuda.max_memory_allocated(device) / (1024 ** 2)} MB")

    dist.all_gather_object(all_local_labels, torch.tensor(local_all_labels, dtype=torch.int64))
    dist.all_gather_object(all_local_preds, torch.tensor(local_all_preds, dtype=torch.int64))
    dist.all_gather_object(all_c2t_acc, torch.tensor([local_c2t_acc], dtype=torch.float32))
    
    # print(f"after all_gather A: Current memory allocated on {device}: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB") 
    # print(f"after all_gather A: Max memory allocated on {device}: {torch.cuda.max_memory_allocated(device) / (1024 ** 2)} MB")
    for i in range(REC_NUM):
        dist.all_gather_object(all_c2t_rec[i], torch.tensor([local_c2t_rec[i]], dtype=torch.float32))
    # dist.all_gather(all_c2t_rec, torch.tensor([local_c2t_rec], dtype=torch.float32).to(device))
    dist.all_gather_object(all_allcnt, torch.tensor([local_allcnt], dtype=torch.int64))
    
    for alpha in alpha_list:
        dist.all_gather_object(all_local_preds_rerank_dict[alpha], torch.tensor(local_all_preds_rerank_dict[alpha], dtype=torch.int64))
        dist.all_gather_object(all_c2t_rerank_acc_list[alpha], torch.tensor(local_c2t_rerank_acc_list[alpha], dtype=torch.float32))
        dist.all_gather_object(all_c2t_rerank_rec_list[alpha], torch.tensor(local_c2t_rerank_rec_list[alpha], dtype=torch.float32))
    
    # Concatenate all gathered results
    if self_rank == 0:
        # 将所有tensor统一放到当前device上
        all_local_labels = [x.to(device) for x in all_local_labels]
        all_local_preds = [x.to(device) for x in all_local_preds]
        all_c2t_acc = [x.to(device) for x in all_c2t_acc]
        for i in range(REC_NUM):
            all_c2t_rec[i] = [x.to(device) for x in all_c2t_rec[i]]
        all_allcnt = [x.to(device) for x in all_allcnt]
        for alpha in alpha_list:
            all_local_preds_rerank_dict[alpha] = [x.to(device) for x in all_local_preds_rerank_dict[alpha]]
            all_c2t_rerank_acc_list[alpha] = [x.to(device) for x in all_c2t_rerank_acc_list[alpha]]
            all_c2t_rerank_rec_list[alpha] = [x.to(device) for x in all_c2t_rerank_rec_list[alpha]]


        all_labels = torch.cat(all_local_labels).cpu().tolist()
        all_preds = torch.cat(all_local_preds).cpu().tolist()
        all_preds_rerank_dict = {alpha: torch.cat(all_local_preds_rerank_dict[alpha]).cpu().tolist() for alpha in alpha_list}
        all_c2t_rerank_acc_list = {alpha: sum(all_c2t_rerank_acc_list[alpha]).tolist() for alpha in alpha_list}
        all_c2t_rerank_rec_list = {alpha: torch.sum(torch.stack(all_c2t_rerank_rec_list[alpha]), dim=0).tolist() for alpha in alpha_list}
        
        all_allcnt = sum(all_allcnt)
        all_allcnt = all_allcnt.item()
        all_c2t_acc = sum(all_c2t_acc)
        all_c2t_acc = all_c2t_acc.item()

        for i in range(REC_NUM):
            all_c2t_rec[i] = sum(all_c2t_rec[i]).item()

        c2t_acc = round(sum([all_c2t_acc]) / sum([all_allcnt]) * 100, 2)
        for i in range(REC_NUM):
            c2t_rec[i] = round(all_c2t_rec[i] / sum([all_allcnt]) * 100, 2)
        c2t_f1 = f1_score(all_labels, all_preds, average='macro') * 100
        c2t_f1 = round(c2t_f1, 2)

        if os.path.exists('data/tabula'):
            cls_result_save_path = f'data/tabula/cls_results_epoch_{alpha}.json'
            cls_results = {
                'all_labels': all_labels,
                'all_preds_rerank_dict': all_preds_rerank_dict,
            }
            with open(cls_result_save_path, 'w') as f:
                json.dump(cls_results, f, indent=4)

        for alpha in alpha_list:
            c2t_rerank_acc_list[alpha] = round(all_c2t_rerank_acc_list[alpha] / sum([all_allcnt]) * 100, 2)
            c2t_rerank_rec_list[alpha] = [round(rec / sum([all_allcnt]) * 100, 2) for rec in all_c2t_rerank_rec_list[alpha]]
            c2t_rerank_f1_list[alpha] = f1_score(all_labels, all_preds_rerank_dict[alpha], average='macro') * 100
            c2t_rerank_f1_list[alpha] = round(c2t_rerank_f1_list[alpha], 2)
            cls_report = readout_zeroshot_report(all_labels, all_preds_rerank_dict[alpha])
            print(f"{alpha=}: ")
            print(cls_report)

        return c2t_acc, c2t_f1, c2t_rec, c2t_rerank_acc_list, c2t_rerank_f1_list, c2t_rerank_rec_list
    else:
        return None, None, None, None, None, None


@torch.no_grad()
def readout_zeroshot_report(all_labels, all_preds):
    cls_report = ''
    for row in confusion_matrix(all_labels, all_preds):
        cls_report += ''.join(f'{str(s):<6}' for s in row) + '\n'
    cls_report += '\n'
    cls_report += classification_report(all_labels, all_preds, digits=4)
    return cls_report