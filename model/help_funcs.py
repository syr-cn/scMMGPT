from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import Levenshtein
from tqdm import tqdm
from typing import List
import numpy as np
import torch

def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    levenshtein = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

        ls = Levenshtein.ratio(gt, out)  # 返回 [0, 1] 之间的相似度
        levenshtein.append(ls)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    ls_score = np.mean(levenshtein)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score, ls_score


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def calculate_topn_accuracy(
    preds_list: List[torch.Tensor], 
    trues_list: List[torch.Tensor], 
    n: int = 10
) -> float:
    """
    计算基因表达预测的 Top-N 准确度。
    
    Args:
        preds_list (list[torch.Tensor]): 预测值列表，每个张量形状为 [batch_size, num_genes]
        trues_list (list[torch.Tensor]): 真实值列表，每个张量形状为 [batch_size, num_genes]
        n (int, optional): Top-N 的 N 值，默认为 10
    
    Returns:
        float: Top-N 准确度（命中数 / 总候选数）
    
    Example:
        >>> preds = [torch.tensor([[0.1, 0.5, 0.3, 0.7], [0.8, 0.2, 0.4, 0.6]])]
        >>> trues = [torch.tensor([[0.5, 0.7, 0.3, 0.1], [0.2, 0.6, 0.8, 0.4]])]
        >>> calculate_topn_accuracy(preds, trues, n=2)
        0.25
    """
    total_hits = 0
    total_num = 0
    
    for pred, true in zip(preds_list, trues_list):
        # 验证输入形状一致性
        assert pred.shape == true.shape, "预测值和真实值形状不一致"
        batch_size, num_genes = pred.shape
        
        # 获取每个样本的真实值和预测值的 Top-N 索引
        _, true_topk = torch.topk(true, k=n, dim=-1)  # [batch_size, n]
        _, pred_topk = torch.topk(pred, k=n, dim=-1)  # [batch_size, n]
        
        # 计算交集数量（利用广播加速）
        true_expanded = true_topk.unsqueeze(2)  # [batch_size, n, 1]
        pred_expanded = pred_topk.unsqueeze(1)  # [batch_size, 1, n]
        matches = (true_expanded == pred_expanded).any(dim=2)  # [batch_size, n]
        
        # 统计命中数和总候选数
        total_hits += matches.sum().item()
        total_num += batch_size
    
    accuracy = total_hits / (total_num * n) if total_num > 0 else 0.0
    return accuracy, total_num

def compute_merged_topn_acc(
    top_n_list, 
    total_num_list,
) -> float:
    """
    合并多卡 Top-N 准确率（加权平均）
    
    Args:
        top_n_list (list[float]): 各卡的 Top-N 准确率列表（每卡一个值，范围 [0,1]）
        total_num_list (list[int]): 各卡对应的样本数量列表（元素为整数）
        
    Returns:
        float: 合并后的全局 Top-N 准确率
    
    Example:
        >>> top1_list = [0.6, 0.5, 0.6, 0.8]
        >>> total_num_list = [32, 32, 30, 30]
        >>> compute_merged_topn_acc(top1_list, total_num_list)
    """
    # 输入验证
    assert len(top_n_list) == len(total_num_list), "两个列表长度必须一致"
    assert all(0 <= acc <= 1 for acc in top_n_list), "准确率值必须在 [0,1] 范围内"
    assert all(num >= 0 for num in total_num_list), "样本数量不能为负数"
    
    total_correct = 0.0
    total_samples = sum(total_num_list)
    
    if total_samples == 0:
        return 0.0  # 避免除以零
    
    # 计算加权准确率
    for acc, num in zip(top_n_list, total_num_list):
        total_correct += acc * num
    
    return total_correct / total_samples


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def align_features(train_data, test_data):
    """将不同基因ID的数据对齐到统一特征空间"""
    # 收集所有基因ID
    all_genes = set()
    for sample in train_data + test_data:
        all_genes.update(sample['gene_ids'])
    gene_list = sorted(all_genes)
    
    # 创建基因索引映射
    gene_idx = {gene: i for i, gene in enumerate(gene_list)}
    
    # 转换特征到统一维度
    def convert(samples):
        features = np.zeros((len(samples), len(gene_list)), dtype=np.float32)
        for i, sample in enumerate(samples):
            for gene, val in zip(sample['gene_ids'], sample['values']):
                if gene in gene_idx:
                    features[i, gene_idx[gene]] = val
        return features
    
    return convert(train_data), convert(test_data)

def knn_pipeline(train_data, train_labels, test_data, test_labels, k=5):
    # 数据对齐
    X_train, X_test = align_features(train_data, test_data)
    
    # 训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, train_labels)
    
    # 预测并评估
    pred_labels = knn.predict(X_test)
    accuracy = accuracy_score(test_labels, pred_labels)
    
    return pred_labels, accuracy

# # 示例数据测试
# if __name__ == "__main__":
#     # 生成示例数据
#     train_examples = [
#         {'gene_ids': np.array([1,3,5]), 'values': np.array([0.5, 0.7, 0.2])},
#         {'gene_ids': np.array([2,4,6]), 'values': np.array([0.6, 0.8, 0.3])}
#     ]
#     train_labels = np.array([0, 1])

#     test_examples = [
#         {'gene_ids': np.array([1,2,7]), 'values': np.array([0.5, 0.6, 0.9])}
#     ]
#     test_labels = np.array([0])

#     # 运行完整流程
#     predictions, acc = knn_pipeline(
#         train_examples, train_labels,
#         test_examples, test_labels,
#         k=1
#     )
    
#     print(f"预测标签: {predictions}")
#     print(f"准确率: {acc*100:.1f}%")