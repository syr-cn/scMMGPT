import os
from typing import Any, Dict
import torch
# from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
# from model.blip2_t5 import Blip2T5
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import caption_evaluate, AttrDict, calculate_topn_accuracy, compute_merged_topn_acc
from transformers import Adafactor


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


# def load_ignore_mismatch(model, state_dict):
#     keys = set(model.state_dict().keys())
#     extra_keys = set()
#     for key in state_dict:
#         if key not in keys:
#             extra_keys.add(key)
#     missing_keys = set()
#     for key in keys:
#         if key not in state_dict:
#             missing_keys.add(key)
#     ## try to print keys that are not included
#     model.load_state_dict(state_dict, strict=False)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict
# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class Blip2Stage2(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.llm_tune != 'full':
            to_be_removed = []
            for key in checkpoint['state_dict']:
                if key.startswith('blip2opt.opt_model') or key.startswith('blip2opt.llm_model'):
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        if isinstance(self.args.save_every_n_epochs, int) and self.args.save_every_n_epochs > 0:
            if self.llm_tune == 'lora' and (self.current_epoch + 1) % self.args.save_every_n_epochs == 0:
                if self.local_rank == 0: # manually fix a bug in peft module
                    if self.args.peft_config:
                        peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                    else:
                        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_dropout)
                    if hasattr(self.blip2lm, 'opt_model'):
                        self.blip2lm.opt_model.peft_config['default'] = peft_config
                        self.blip2lm.opt_model.save_pretrained(os.path.join(self.logger.save_dir, f'lora_epoch_{self.current_epoch}'))
                    elif hasattr(self.blip2lm, 'llm_model'):
                        self.blip2lm.llm_model.peft_config['default'] = peft_config
                        self.blip2lm.llm_model.save_pretrained(os.path.join(self.logger.save_dir, f'lora_epoch_{self.current_epoch}'))
        return super().on_save_checkpoint(checkpoint)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        if args.opt_model.find('llama') >= 0 or args.opt_model.find('Llama') >= 0:
            self.blip2lm = Blip2Llama(args.bert_name, 
                                      args.tune_gene_encoder, 
                                      args.num_query_token, 
                                      args.cross_attention_freq, 
                                      args.llm_tune, 
                                      args.peft_dir, 
                                      args.opt_model, 
                                      args.prompt,
                                      args)
        else:
            raise NotImplementedError()
        
        # if len(args.stage1_path) > 0:
        #     self.load_from_stage1_checkpoint(args.stage1_path)

        self.tokenizer = self.blip2lm.init_tokenizer(args.bert_name)
        self.save_hyperparameters(args)

        self.test_predictions = []
        self.test_values1p = []
        self.test_targets = []
        self.test_target_values = []
        self.test_cell_types = []
        self.test_gene_ids = []
        
        self.val_predictions = []
        self.val_values1p = []
        self.val_targets = []
        self.val_target_values = []
        self.val_cell_types = []
        self.val_gene_ids = []

    def load_from_stage1_checkpoint(self, path, llm_model):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        cell_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.cell_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_gene_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_gene')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')

        # 修复键名不匹配问题：将 Wqkv 替换为 in_proj
        if "tinyllama" in llm_model.lower():
            new_cell_encoder_dict = {}
            for key, value in cell_encoder_dict.items():
                new_key = key.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
                new_key = new_key.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
                new_cell_encoder_dict[new_key] = value
            cell_encoder_dict = new_cell_encoder_dict

        load_ignore_unexpected(self.blip2lm.Qformer, qformer_dict)
        self.blip2lm.cell_encoder.load_state_dict(cell_encoder_dict)
        self.blip2lm.ln_gene.load_state_dict(ln_gene_dict)
        self.blip2lm.query_tokens.data.copy_(qs_weight)
        return self
    
    # def load_from_stage1_checkpoint(self, path):
    #     ckpt = torch.load(path, map_location='cpu')
    #     state_dict = ckpt['state_dict']
    #     state_dict = {k[13:]: v for k,v in state_dict.items()}
    #     load_ignore_mismatch(self.blip2lm, state_dict)
    #     return self
    
    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
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

    # def test_epoch_end(self, outputs):
    #     list_predictions, list_targets = zip(*outputs)
    #     predictions = [i for ii in list_predictions for i in ii]
    #     targets = [i for ii in list_targets for i in ii]

    #     all_predictions = [None for _ in range(self.trainer.world_size)]
    #     all_targets = [None for _ in range(self.trainer.world_size)]

    #     dist.all_gather_object(all_predictions, predictions)
    #     dist.all_gather_object(all_targets, targets)
    #     if self.global_rank == 0:
    #         all_predictions = [i for ii in all_predictions for i in ii]
    #         all_targets = [i for ii in all_targets for i in ii]
    #         self.save_predictions(all_predictions, all_targets)
    #         ## fixme: I am not sure if the max length is the same as previous experiments
    #         bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
    #             caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
    #         self.log("bleu2", bleu2, sync_dist=False)
    #         self.log("bleu4", bleu4, sync_dist=False)
    #         self.log("rouge_1", rouge_1, sync_dist=False)
    #         self.log("rouge_2", rouge_2, sync_dist=False)
    #         self.log("rouge_l", rouge_l, sync_dist=False)
    #         self.log("meteor_score", meteor_score, sync_dist=False)

    def save_predictions(self, predictions, targets):
        assert len(predictions) == len(targets)
        with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        text_tokens, gene_inputs, texts = batch
        predictions = self.blip2lm.generate(
                batch=batch, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
       
        self.test_predictions.append(predictions["text"])
        self.test_values1p.append(predictions['values1p'])
        self.test_targets.append(texts)
        self.test_gene_ids.append(gene_inputs['gene_ids'])
        self.test_target_values.append(gene_inputs['values'])
        self.test_cell_types.append(gene_inputs['cell_type'])

    def on_test_epoch_end(self):
        # 展平列表时将张量转移到CPU并转换为Python对象
        def flatten_to_cpu(data_list):
            result = []
            for batch in data_list:
                if isinstance(batch, torch.Tensor):
                    batch = batch.detach().cpu()
                if isinstance(batch, list):
                    result.extend([i.cpu() if isinstance(i, torch.Tensor) else i for i in batch])
                else:
                    result.append(batch)
            return result

        # 处理所有需要收集的数据
        predictions = flatten_to_cpu([i for ii in self.test_predictions for i in ii])
        targets = flatten_to_cpu([i for ii in self.test_targets for i in ii])
        cell_types = flatten_to_cpu([i for ii in self.test_cell_types for i in ii])
        gene_ids = flatten_to_cpu([i for ii in self.test_gene_ids for i in ii])
        target_values = flatten_to_cpu([i for ii in self.test_target_values for i in ii])
        values1p = flatten_to_cpu([i for ii in self.test_values1p for i in ii])

        # 创建收集容器（保持CPU数据）
        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_cell_types = [None for _ in range(self.trainer.world_size)]
        all_gene_ids = [None for _ in range(self.trainer.world_size)]
        all_target_values = [None for _ in range(self.trainer.world_size)]
        all_values1p = [None for _ in range(self.trainer.world_size)]

        # 执行收集（确保数据是纯Python对象）
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        dist.all_gather_object(all_cell_types, cell_types)
        dist.all_gather_object(all_gene_ids, gene_ids)
        dist.all_gather_object(all_target_values, target_values)
        dist.all_gather_object(all_values1p, values1p)

        if self.global_rank == 0:
            # 展平收集结果（所有数据保持CPU状态）
            def merge_all(data):
                return [item for sublist in data if sublist is not None for item in (sublist if isinstance(sublist, list) else [sublist])]
            
            all_predictions = merge_all(all_predictions)
            all_targets = merge_all(all_targets)
            all_cell_types = merge_all(all_cell_types)
            all_gene_ids = merge_all(all_gene_ids)
            all_target_values = merge_all(all_target_values)
            all_values1p = merge_all(all_values1p)

            # 保存结果到文件（全程使用CPU）
            prediction_path = self.logger.log_dir
            os.makedirs(prediction_path, exist_ok=True)
            
            with open(os.path.join(prediction_path, 'predictions.txt'), 'w', encoding='utf8') as ff:
                for i in range(len(all_predictions)):
                    ff.write(f"=== {i:6} ===\n")
                    ff.write(str(all_predictions[i]) + '\n')
                    ff.write(str(all_targets[i]) + '\n')

            # 保存为CPU数据（使用pickle替代torch.save可进一步降低显存）
            save_dict = {
                "all_predictions": all_predictions,
                "all_targets": all_targets,
                "all_cell_types": all_cell_types,
                "all_gene_ids": all_gene_ids,
                "all_target_values": all_target_values,
                "all_values1p": all_values1p
            }
            
            # 使用pickle保存避免torch.save可能的显存峰值
            import pickle
            with open(os.path.join(prediction_path, 'test_dict.pkl'), 'wb') as f:
                pickle.dump(save_dict, f)

            # 计算指标（确保评估函数不产生显存占用）
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score, ls_score = \
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
            
            # 记录指标（sync_dist=False避免不必要的通信）
            metrics = {
                "bleu2": bleu2,
                "bleu4": bleu4,
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
                "meteor_score": meteor_score,
                "ls_score": ls_score
            }
            for name, value in metrics.items():
                self.log(name, value, sync_dist=False, prog_bar=True)

        # 清空缓存释放显存
        torch.cuda.empty_cache()
        # 清空临时存储
        self.test_predictions.clear()
        self.test_targets.clear()
        self.test_cell_types.clear()
        self.test_gene_ids.clear()
        self.test_target_values.clear()
        self.test_values1p.clear()

    # def on_test_epoch_end(self):
    #     predictions = [i for ii in self.test_predictions for i in ii]
    #     targets = [i for ii in self.test_targets for i in ii]
        
    #     all_predictions = [None for _ in range(self.trainer.world_size)]
    #     all_targets = [None for _ in range(self.trainer.world_size)]

    #     dist.all_gather_object(all_predictions, predictions)
    #     dist.all_gather_object(all_targets, targets)
        
    #     cell_types = [i for ii in self.test_cell_types for i in ii]
    #     gene_ids = [i for ii in self.test_gene_ids for i in ii]
    #     target_values = [i for ii in self.test_target_values for i in ii]
    #     values1p = [i for ii in self.test_values1p for i in ii]
    #     # print(f"rank = {self.global_rank}, {len(cell_types)=}, {cell_types[0]=}")
        
    #     all_cell_types = [None for _ in range(self.trainer.world_size)]
    #     all_gene_ids = [None for _ in range(self.trainer.world_size)]
    #     all_target_values = [None for _ in range(self.trainer.world_size)]
    #     all_values1p = [None for _ in range(self.trainer.world_size)]

    #     dist.all_gather_object(all_cell_types, cell_types)
    #     dist.all_gather_object(all_gene_ids, gene_ids)
    #     dist.all_gather_object(all_target_values, target_values)
    #     dist.all_gather_object(all_values1p, values1p)

    #     if self.global_rank == 0:
    #         all_predictions = [i for ii in all_predictions for i in ii]
    #         all_targets = [i for ii in all_targets for i in ii]
    #         all_cell_types = [i for ii in all_cell_types for i in ii]
    #         all_gene_ids = [i for ii in all_gene_ids for i in ii]
    #         all_target_values = [i for ii in all_target_values for i in ii]
    #         all_values1p = [i for ii in all_values1p for i in ii]
    #         prediction_path = self.logger.log_dir

    #         with open(os.path.join(prediction_path, 'predictions.txt'), 'w', encoding='utf8') as ff:
    #             for i in range(len(all_predictions)):
    #                 ff.write(f"=== {i:6} ===\n")
    #                 ff.write(all_predictions[i] + '\n')
    #                 ff.write(all_targets[i] + '\n')

    #         save_dict = {
    #             "all_predictions" : all_predictions,
    #             "all_targets": all_targets,
    #             "all_cell_types": all_cell_types,
    #             "all_gene_ids": all_gene_ids,
    #             "all_target_values": all_target_values,
    #             "all_values1p": all_values1p
    #         }

    #         torch.save(save_dict, os.path.join(prediction_path, 'test_dict.pt'))

    #         bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score, ls_score = \
    #             caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
    #         self.log("bleu2", bleu2, sync_dist=False)
    #         self.log("bleu4", bleu4, sync_dist=False)
    #         self.log("rouge_1", rouge_1, sync_dist=False)
    #         self.log("rouge_2", rouge_2, sync_dist=False)
    #         self.log("rouge_l", rouge_l, sync_dist=False)
    #         self.log("meteor_score", meteor_score, sync_dist=False)
    #         self.log("ls_score", ls_score, sync_dist=False)

    #     # all_values1p = [None for _ in range(self.trainer.world_size)]
    #     # all_target_values = [None for _ in range(self.trainer.world_size)

    #     n_list = [1, 3, 5, 7, 9]  # 需要计算的 Top-N 列表
    #     for n in n_list:
    #         # 计算当前N值的准确率和样本数
    #         topn_acc, total_num = calculate_topn_accuracy(self.test_values1p, self.test_target_values, n)
            
    #         # 分布式收集各卡结果
    #         all_topn_acc = [None for _ in range(self.trainer.world_size)]
    #         all_num_list = [None for _ in range(self.trainer.world_size)]
    #         dist.all_gather_object(all_topn_acc, topn_acc)
    #         dist.all_gather_object(all_num_list, total_num)
            
    #         # 仅 rank 0 计算合并结果并记录日志
    #         if self.global_rank == 0:
    #             merged_acc = compute_merged_topn_acc(all_topn_acc, all_num_list)
    #             self.log(f"top{n}_acc", merged_acc, sync_dist=False)
        # top1_acc, total_num = calculate_topn_accuracy(self.test_values1p, self.test_target_values, 1)
        # all_top1_acc = [None for _ in range(self.trainer.world_size)]
        # all_num_list = [None for _ in range(self.trainer.world_size)]
        # dist.all_gather_object(all_top1_acc, top1_acc)
        # dist.all_gather_object(all_num_list, total_num)
        # # dist.all_gather_object(all_values1p, self.test_values1p)
        # # dist.all_gather_object(all_target_values, self.test_target_values)

        # if self.global_rank == 0:
            
        #     # all_values1p = [i for ii in all_values1p for i in ii]
        #     # all_target_values = [i for ii in all_target_values for i in ii]

        #     # top3_acc = calculate_topn_accuracy(all_values1p, all_target_values, 3)
        #     # top5_acc = calculate_topn_accuracy(all_values1p, all_target_values, 5)
        #     # top10_acc = calculate_topn_accuracy(all_values1p, all_target_values, 10)
        #     top1_acc_final = compute_merged_topn_acc(all_top1_acc, all_num_list)
        #     self.log("top1_acc", top1_acc_final, sync_dist=False)
            # self.log("top3_acc", top3_acc, sync_dist=False)
            # self.log("top5_acc", top5_acc, sync_dist=False)
            # self.log("top10_acc", top10_acc, sync_dist=False)

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     if dataloader_idx == 0:
    #         _, _, prompt_lens = batch
    #         batch_size = prompt_lens.shape[0]
    #         loss = self.blip2lm(batch)
    #         ###============== Overall Loss ===================###
    #         self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
    #         return loss['loss']
    #     elif dataloader_idx == 1:
    #         reaction_tokens, _, _ = batch
    #         batch_size = reaction_tokens.input_ids.shape[0]
    #         loss = self.blip2lm.forward_reaction(batch)
    #         ###============== Overall Loss ===================###
    #         self.log("val reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
    #         return loss['loss']
    #     else:
    #         raise NotImplementedError
    
    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     if dataloader_idx == 0:
    #         text_tokens, _ = batch
    #         batch_size = text_tokens["input_ids"].shape[0]
    #         loss = self.blip2lm(batch)
    #         self.log("loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
    #         self.log("lm_loss", float(loss['lm_loss']), batch_size=batch_size, sync_dist=True, prog_bar=True)
    #         self.log("rec_loss", float(loss['rec_loss']), batch_size=batch_size, sync_dist=True, prog_bar=True)
    #         self.log("expr_loss", float(loss['expr_loss']), batch_size=batch_size, sync_dist=True, prog_bar=True)
            
    #         return loss['loss']
    #     elif dataloader_idx == 1:
            
    #         text_tokens, gene_inputs, texts = batch
    #         predictions = self.blip2lm.generate(
    #             batch=batch, 
    #             do_sample=self.do_sample,
    #             num_beams=self.num_beams,
    #             max_length=self.max_len,
    #             min_length=self.min_len
    #         )
        
    #         self.val_predictions.append(predictions["text"])
    #         self.val_values1p.append(predictions['values1p'])
    #         self.val_targets.append(texts)
    #         self.val_gene_ids.append(gene_inputs['gene_ids'])
    #         self.val_target_values.append(gene_inputs['values'])
    #         self.val_cell_types.append(gene_inputs['cell_type'])
    #     else:
    #         raise NotImplementedError


    # def on_validation_epoch_end(self):
    #     predictions = [i for ii in self.val_predictions for i in ii]
    #     targets = [i for ii in self.val_targets for i in ii]
        
    #     all_predictions = [None for _ in range(self.trainer.world_size)]
    #     all_targets = [None for _ in range(self.trainer.world_size)]

    #     dist.all_gather_object(all_predictions, predictions)
    #     dist.all_gather_object(all_targets, targets)
        
    #     cell_types = [i for ii in self.val_cell_types for i in ii]
    #     gene_ids = [i for ii in self.val_gene_ids for i in ii]
    #     target_values = [i for ii in self.val_target_values for i in ii]
    #     values1p = [i for ii in self.val_values1p for i in ii]
    #     # print(f"rank = {self.global_rank}, {len(cell_types)=}, {cell_types[0]=}")
        
    #     all_cell_types = [None for _ in range(self.trainer.world_size)]
    #     all_gene_ids = [None for _ in range(self.trainer.world_size)]
    #     all_target_values = [None for _ in range(self.trainer.world_size)]
    #     all_values1p = [None for _ in range(self.trainer.world_size)]

    #     dist.all_gather_object(all_cell_types, cell_types)
    #     dist.all_gather_object(all_gene_ids, gene_ids)
    #     dist.all_gather_object(all_target_values, target_values)
    #     dist.all_gather_object(all_values1p, values1p)

    #     self.val_predictions = []
    #     self.val_values1p = []
    #     self.val_targets = []
    #     self.val_target_values = []
    #     self.val_cell_types = []
    #     self.val_gene_ids = []

    #     if self.global_rank == 0:
    #         all_predictions = [i for ii in all_predictions for i in ii]
    #         all_targets = [i for ii in all_targets for i in ii]
    #         all_cell_types = [i for ii in all_cell_types for i in ii]
    #         all_gene_ids = [i for ii in all_gene_ids for i in ii]
    #         all_target_values = [i for ii in all_target_values for i in ii]
    #         all_values1p = [i for ii in all_values1p for i in ii]
    #         prediction_path = self.logger.log_dir

    #         step = self.trainer.global_step

    #         with open(os.path.join(prediction_path, f'val_predictions_{step}.txt'), 'w', encoding='utf8') as ff:
    #             for i in range(len(all_predictions)):
    #                 ff.write(f"=== {i:6} ===\n")
    #                 ff.write(all_predictions[i] + '\n')
    #                 ff.write(all_targets[i] + '\n')

    #         save_dict = {
    #             "all_predictions" : all_predictions,
    #             "all_targets": all_targets,
    #             "all_cell_types": all_cell_types,
    #             "all_gene_ids": all_gene_ids,
    #             "all_target_values": all_target_values,
    #             "all_values1p": all_values1p
    #         }

    #         torch.save(save_dict, os.path.join(prediction_path, f'val_dict_{step}.pt'))

    #         bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score, ls_score = \
    #             caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
    #         self.log("bleu2", bleu2, sync_dist=False)
    #         self.log("bleu4", bleu4, sync_dist=False)
    #         self.log("rouge_1", rouge_1, sync_dist=False)
    #         self.log("rouge_2", rouge_2, sync_dist=False)
    #         self.log("rouge_l", rouge_l, sync_dist=False)
    #         self.log("meteor_score", meteor_score, sync_dist=False)
    #         self.log("ls_score", ls_score, sync_dist=False)
        

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[0]["input_ids"].size(0)
        ###============== Overall Loss ===================###
        loss = self.blip2lm(batch)
        self.log("loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lm_loss", float(loss['lm_loss']), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("rec_loss", float(loss['rec_loss']), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("expr_loss", float(loss['expr_loss']), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True, prog_bar=True)
        return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gene_encoder', action='store_true', default=False)
        parser.add_argument('--tune_Qformer', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=256)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        return parent_parser


