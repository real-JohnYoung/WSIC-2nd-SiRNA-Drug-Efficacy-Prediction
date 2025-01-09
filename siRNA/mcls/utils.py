import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
import RNA 
from itertools import product
import math
from collections import Counter
import pandas as pd
from Bio.Seq import Seq
from Bio import pairwise2
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果您使用的是多个GPU
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 避免卷积算法在训练时的选择波动


#———————————————————————————tokenizer—————————————————————————————#
class GenomicTokenizer:
    def __init__(self, ngram=5, stride=2):
        self.ngram = ngram
        self.stride = stride
        
    def tokenize(self, t):
        t = t.upper()
        if self.ngram == 1:
            toks = list(t)
        else:
            toks = [t[i:i+self.ngram] for i in range(0, len(t), self.stride) if len(t[i:i+self.ngram]) == self.ngram]
        if len(toks[-1]) < self.ngram:
            toks = toks[:-1]
        return toks


class GenomicVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        
    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(tokens)
        itos = ['<pad>'] + [o for o,c in freq.most_common(max_vocab-1) if c >= min_freq]
        return cls(itos)


def calculate_metrics(y_true, y_pred, threshold=30):
    mae = np.mean(np.abs(y_true - y_pred))

    y_true_binary = (y_true < threshold).astype(int) 
    y_pred_binary = (y_pred < threshold).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100

    precision = precision_score(y_true_binary, y_pred_binary, average='binary')
    recall = recall_score(y_true_binary, y_pred_binary, average='binary')
    
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return score


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for seqs, cat_data, num_data,prior, target in test_loader:
            seqs = [x.to(device) for x in seqs]
            cat_data = [x.to(device) for x in cat_data]
            num_data = num_data.to(device)
            prior = prior.to(device)
            seq_feat,prediction = model(seqs, cat_data, num_data,prior)
            predictions.extend(prediction.cpu().numpy())
            targets.extend(target.numpy())

    y_pred = np.array(predictions)
    y_pred = np.clip(y_pred, 0, 100)##not very reasonable
    y_test = np.array(targets)
    
    score = calculate_metrics(y_test, y_pred)
    print(f"Test Score: {score:.4f}")



class AffinityNetwork(nn.Module):
    def __init__(self, input_dim):
        super(AffinityNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, antisense_feat, mrna_feat):
        combined_feat = torch.cat([antisense_feat, mrna_feat], dim=1)
        x = F.relu(self.fc1(combined_feat))
        affinity = F.sigmoid(self.fc2(x))
        return affinity
    

class AffinityNetwork(nn.Module):
    def __init__(self, input_dim):
        super(AffinityNetwork, self).__init__()
        # 定义三层全连接网络用于亲和力计算
        self.fc1 = nn.Linear(input_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, antisense_feat, mrna_feat):
        # 将siRNA和mRNA特征进行拼接
        combined_feat = torch.cat([antisense_feat, mrna_feat], dim=1)
        x = F.relu(self.fc1(combined_feat))
        affinity = F.sigmoid(self.fc2(x))
        return affinity

class CustomLoss(nn.Module):
    def __init__(self, 
                 input_dim=256,
                 task1_weight=0.5,
                 task2_weight=0.75,
                 task3_weight=1, weight_0_30=2, weight_above_30=1, tolerance=10):
        super(CustomLoss, self).__init__()

        # 任务1：亲和力对比学习网络
        self.task1_weight = task1_weight
        self.affinity_network1 = AffinityNetwork(input_dim)
        
        # 任务2：修饰siRNA亲和力对比网络
        self.task2_weight = task2_weight
        self.affinity_network2 = AffinityNetwork(input_dim)

        # 任务3：留存值预测任务
        self.task3_weight = task3_weight
        self.weight_0_30 = weight_0_30
        self.weight_above_30 = weight_above_30
        self.tolerance = tolerance
        self.loss = nn.MSELoss(reduction='none') 

    def task1_forward(self, antisense_feat, mrna_feat):
        # 计算A样本的亲和力
        positive_affinity = self.affinity_network1(antisense_feat, mrna_feat)
        # 随机取其他样本作为负样本计算亲和力
        negative_affinity = self.affinity_network1(torch.roll(antisense_feat, shifts=1, dims=0), mrna_feat)
        # 三元组损失，期望positive大于negative
        margin = 0.3
        loss = F.relu(negative_affinity - positive_affinity + margin).mean()

        return loss
    
    def task2_forward(self, antisense_feat, modified_antisense_feat, mrna_feat):
        # 计算不带修饰的siRNA亲和力
        normal_affinity = self.affinity_network2(antisense_feat, mrna_feat)
        # 计算带修饰的siRNA亲和力
        modified_affinity = self.affinity_network2(modified_antisense_feat, mrna_feat)
        
        # 期望modified的亲和力高于normal
        margin = 0.2
        loss = F.relu(normal_affinity - modified_affinity + margin).mean()
        return loss

    def task3_forward(self, predictions, targets):
        loss = torch.zeros_like(targets)
        mask_0_30 = targets < 30
        mask_above_30 = targets >= 30
        
        # 计算基本损失
        base_loss_0_30 = self.loss(predictions[mask_0_30], targets[mask_0_30])
        base_loss_above_30 = self.loss(predictions[mask_above_30], targets[mask_above_30])
        
        # 应用误差容忍度
        mask_within_tolerance_0_30 = torch.abs(predictions[mask_0_30] - targets[mask_0_30]) < self.tolerance*0.5
        mask_within_tolerance_above_30 = torch.abs(predictions[mask_above_30] - targets[mask_above_30]) < self.tolerance
        
        base_loss_0_30[mask_within_tolerance_0_30] = 0
        base_loss_above_30[mask_within_tolerance_above_30] = 0
        
        # 计算加权损失
        loss[mask_0_30] = self.weight_0_30 * base_loss_0_30
        loss[mask_above_30] = self.weight_above_30 * base_loss_above_30
        
        return loss.mean()

    def forward(self, seq_feat, predictions, targets):
        # 任务1的损失
        task1_loss = self.task1_forward(seq_feat['antisense_feat'], seq_feat['mrna_feat'])
        # 任务2的损失
        task2_loss = self.task2_forward(seq_feat['antisense_feat'], seq_feat['modified_antisense_feat'], seq_feat['mrna_feat'])
        # 任务3的损失
        task3_loss =self.task3_forward(predictions, targets)
        # 总损失
        epsilon = 1e-3
        total_loss = self.task1_weight * task1_loss + self.task2_weight * task2_loss +  self.task3_weight * task3_loss
        return total_loss
    



#————————————————————————————————————————————————————————————————————————————————get priors————————————————————————————————————————————————————————————————————————————————————————————————————————————#
    
def gibbs_energy(seq):
    energy_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 
    'table': np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}

    result = []
    for i in range(len(seq)-1):
        index_1 = energy_dict.get(seq[i])
        index_2 = energy_dict.get(seq[i + 1])
        result.append(energy_dict['table'][index_1, index_2])

    result.append(np.array(result).sum().round(3))
    result.append((result[0] - result[-2]).round(3)) 
    result = result[:24] + (24-len(result))*[0]
    result = np.array(result)
    return result

def get_gc_sterch(seq):
    max_len, tem_len = 0, 0
    for i in range(len(seq)):
        if seq[i] == 'G' or seq[i] == 'C':
            tem_len += 1
            max_len = max(max_len, tem_len)
        else:
            tem_len = 0

    result = round((max_len / len(seq)), 3)
    return np.array([result])

def get_gc_percentage(seq):
    result = round(((seq.count('C') + seq.count('G')) / len(seq)), 3)
    return result

def get_gc_features(seq):
    overall_gc = get_gc_percentage(seq)
    region1_gc = get_gc_percentage(seq[1:7])
    region2_gc = get_gc_percentage(seq[8:18])
    energy_valley = (region1_gc < 0.3) and (region2_gc < 0.3)
    return np.array([overall_gc, region1_gc, region2_gc, energy_valley])


def get_single_comp_percent(seq):
    nt_percent = []
    for base_i in list(['A', 'G', 'C', 'U']):
        nt_percent.append(round((seq.count(base_i) / len(seq)), 3))
    return np.array(nt_percent)

def get_di_comp_percent(seq):
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / (len(seq) - 1)), 3))
    return np.array(di_nt_percent)

def get_tri_comp_percent(seq): 
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=3))
    tri_nt_percent = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0] + pmt_i[1] + pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt) / (len(seq) - 2)), 3))
    return np.array(tri_nt_percent)

def secondary_struct(seq):
    def _percentage(if_paired):
        paired_percent = (if_paired.count('(') + if_paired.count(')')) / len(if_paired)
        unpaired_percent = (if_paired.count('.')) / len(if_paired)
        return [paired_percent, unpaired_percent]

    paired_seq, min_free_energy = RNA.fold(seq)
    return np.array(_percentage(paired_seq)+[min_free_energy])

def create_pssm(train_seq):
    train_seq = [list(seq.upper())[:19] for seq in train_seq]
    train_seq = np.array(train_seq)

    nr, nc = np.shape(train_seq)
    pseudocount = nr ** 0.5 
    pssm = []
    for c in range(0, nc):
        col_c = train_seq[:, c].tolist()
        f_A = round(((col_c.count('A') + pseudocount) / (nr + pseudocount)), 3)
        f_G = round(((col_c.count('G') + pseudocount) / (nr + pseudocount)), 3)
        f_C = round(((col_c.count('C') + pseudocount) / (nr + pseudocount)), 3)
        f_U = round(((col_c.count('U') + pseudocount) / (nr + pseudocount)), 3)
        pssm.append([f_A, f_G, f_C, f_U])
    pssm = np.array(pssm)
    pssm = pssm.transpose()
    return pssm

def score_seq_by_pssm(pssm, seq): 
    nt_order = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    ind_all = list(range(0, 19))
    scores = [pssm[nt_order[nt], i] for nt, i in zip(seq, ind_all)]
    log_score = sum([-math.log2(i) for i in scores])
    return np.array([log_score])




# Helper function to parallelize apply
def parallel_apply(df, func, num_partitions=None, num_workers=None):
    if num_partitions is None:
        num_partitions = mp.cpu_count()  # 默认分区数量等于CPU核心数
    if num_workers is None:
        num_workers = mp.cpu_count()  # 默认工作线程等于CPU核心数
    
    df_split = np.array_split(df, num_partitions)  # 将DataFrame拆分成多部分
    pool = mp.Pool(num_workers)
    
    # 使用tqdm显示进度条
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def calculate_watson_crick_free_energy(sense_seq, antisense_seq):
    """
    计算 Watson-Crick 配对的自由能，忽略 antisense 序列的最后两个碱基（尾巴）
    :param sense_seq: siRNA 的正义链 (sense)
    :param antisense_seq: siRNA 的反义链 (antisense)
    :return: Watson-Crick 自由能
    """
    antisense_seq_trimmed = antisense_seq[:-2]
    cofold = RNA.cofold(sense_seq + "&" + antisense_seq_trimmed)
    
    return np.array([cofold[1]])

# 反义链与gene target序列的匹配特征
def get_alignment_score(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1,seq2)
    best_alignment = alignments[0]
    score = best_alignment[2]
    return np.array([score])


def check_bases(siRNA_seq):
    encoding = np.zeros(13, dtype=int)
    rules = [
        ('A', [1]),        # A(1)
        ('C', [1]),        # C(1)
        ('G', [1]),        # G(1)
        ('U', [19]),       # U(19)
        ('UU', range(1, 11)),  # UU(1-10)
        ('U', [1]),        # U(1)
        ('A', [2]),        # A(2)
        ('CA', range(1, 11)),  # CA(1-10)
        ('G', [19]),       # G(19)
        ('GG', range(10, 20)), # GG(10-19)
        ('CC', range(1, 11)),  # CC(1-10)
        ('G', [7]),        # G(7)
        ('U', [18])        # U(18)
    ]
    
    for i, (base, positions) in enumerate(rules):
        for pos in positions:
            if siRNA_seq[pos - 1:pos - 1 + len(base)] == base:
                encoding[i] = 1
                break
    return np.array(encoding)

# for modified seq
def encode_modifications(seq):
    modifications = []
    for base in seq:
        if base.islower(): 
            modifications.append(1)
        else:
            modifications.append(0)
    modifications = modifications +[0]*10

    return np.array(modifications[:30])



def get_priors_(data):
    data['gc_sterch'] = data['siRNA_antisense_seq'].apply(get_gc_sterch)                    #1 GC含量0~1
    data['gc_features'] = data['siRNA_antisense_seq'].apply(get_gc_features)               # 1 GC含量0~1，
    data['gibbs_energy_result'] = data['siRNA_antisense_seq'].apply(gibbs_energy)           # 24
    data['single_nt_percent'] = data['siRNA_antisense_seq'].apply(get_single_comp_percent)  # 4
    data['di_nt_percent'] = data['siRNA_antisense_seq'].apply(get_di_comp_percent)          # 16
    data['tri_nt_percent'] = data['siRNA_antisense_seq'].apply(get_tri_comp_percent)        # 64
    data['secondary_struct'] = data['siRNA_antisense_seq'].apply(secondary_struct)          # 3
    data['mrna_secondary_struct'] = data['Extended_Sequence'].apply(secondary_struct)       # 3
    data['mrna_features'] = data['Extended_Sequence'].apply(generate_mrna_features)
    data['watson_crick_free_energy'] = data.apply(lambda x: calculate_watson_crick_free_energy(x['siRNA_sense_seq'], x['siRNA_antisense_seq']), axis=1)
    data['encode_modifications'] = data['modified_siRNA_sense_seq'].apply(encode_modifications)
    data['interaction_energy'] = data.apply(lambda x: calculate_interaction_energy(x['siRNA_antisense_seq'], x['Extended_Sequence']), axis=1)
    data['check_bases'] = data['siRNA_antisense_seq'].apply(check_bases)

    return data

# 主函数，使用多线程并行处理
def get_priors(data):
    print('Start parallel computation...')
    data = parallel_apply(data, get_priors_)
    print('Parallel computation completed!')
    return data


def fill_missing_values_with_random_choice(series):
    non_null_values = series.dropna().tolist()
    return series.apply(lambda x: random.choice(non_null_values) if pd.isnull(x) else x)

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def find_best_antisense_match(row):
    gene_sequence = row['gene_target_seq']
    antisense_sequence = row['siRNA_antisense_seq'].replace('U', 'T')
    processed_siRNA_antisense_seq = antisense_sequence[:19]  # Take first 19 bases
    antisense_seq_obj = Seq(processed_siRNA_antisense_seq)
    sense_sequence = str(antisense_seq_obj.reverse_complement())
    seed_region = sense_sequence[1:8]  # Extract seed region from positions 2 to 8

    if not gene_sequence or len(processed_siRNA_antisense_seq) < 19:
        return processed_siRNA_antisense_seq, None, 0, None  # Ensure valid sequence data is present

    max_score = float('-inf')
    best_start = None
    matched_sequence = None
    best_seed_match_score = 0  # Initialize best seed region match score


    # First, search for exact matches of the seed region in the gene sequence
    for i in range(len(gene_sequence) - 18):
        candidate_seed_region = gene_sequence[i+1:i+8]  # Compare to seed region
        if candidate_seed_region == seed_region:
            best_seed_match_score = 14
            candidate_sequence = gene_sequence[i:i + 19]
            alignments = pairwise2.align.localms(sense_sequence, candidate_sequence, 2, -1, -0.5, -0.1, one_alignment_only=True)
            score = alignments[0][2]
            # Apply higher weight to the seed region match
            score += 10  # Arbitrary weight for seed region match, adjust as necessary

            if score > max_score:
                max_score = score
                best_start = i
                matched_sequence = candidate_sequence

    # If no exact match, find the best partial match
    if best_start is None:
        for i in range(len(gene_sequence) - 18):
            candidate_seed_region = gene_sequence[i+1:i+8]  # Compare to seed region
            seed_match_score = sum(1 for a, b in zip(candidate_seed_region, seed_region) if a == b)
            candidate_sequence = gene_sequence[i:i + 19]
            alignments = pairwise2.align.localms(sense_sequence, candidate_sequence, 2, -1, -0.5, -0.1, one_alignment_only=True)
            score = alignments[0][2]
            
            # First prioritize by seed match score, then by overall score
            if (seed_match_score > best_seed_match_score) or (seed_match_score == best_seed_match_score and score > max_score):
                max_score = score
                best_start = i
                matched_sequence = candidate_sequence
                best_seed_match_score = seed_match_score


   
    if best_start is not None:
        extended_start = max(0, best_start - 20)
        extended_end = min(len(gene_sequence), best_start + 19 + 20)
        extended_sequence = gene_sequence[extended_start:extended_end]
    else:
        extended_sequence = None

    return processed_siRNA_antisense_seq, matched_sequence, max_score, extended_sequence, best_seed_match_score

# Function to apply parallel processing using ProcessPoolExecutor
def apply_parallel(data, func):
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, [row for _, row in data.iterrows()]), total=len(data)))
    return results



# 提取上游和下游邻近序列
def get_neighbourhood(seq, target_len=19):
    # 默认先取上下游19个碱基（上下游各19个），如果不足则取实际长度
    matched_start = (len(seq) - target_len) // 2
    matched_seq = seq[matched_start:matched_start + target_len]
    upstream = seq[:matched_start]
    downstream = seq[matched_start + target_len:]
    return upstream, matched_seq, downstream

# 核苷酸百分比计算
def calculate_base_percent(seq, k):
    total_kmers = len(seq) - k + 1
    if total_kmers > 0:
        kmer_counts = Counter([seq[i:i+k] for i in range(total_kmers)])
        return {kmer: round(v / total_kmers, 3) for kmer, v in kmer_counts.items()}
    else:
        return {}

# 生成mRNA相关特征，并返回为向量
def generate_mrna_features(seq, target_len=19):
    upstream, matched_seq, downstream = get_neighbourhood(seq, target_len)

    # 定义要计算的核苷酸组合
    single_nts = ['A', 'U', 'G', 'C']
    di_nts = ['AA', 'AU', 'UA', 'UU', 'GG', 'CG', 'GC', 'CC']
    tri_nts = ['AAU', 'AAA', 'GGG', 'ACU', 'ACA', 'GGC', 'UAG', 'CGU', 'UUA', 'UAU', 'AGC', 'CGG', 
               'UAA', 'GAA', 'GCC', 'GUU', 'CUG', 'UGU', 'ACC', 'CCG', 'CGA', 'UCG', 'AUA']

    features = []

    # 计算全序列、邻近区域和target site的核苷酸百分比
    sequences = {
        'mRNA': seq,
        #'neighbourhood': upstream + downstream,
        'upstream': upstream,
        'downstream': downstream
    }

    # 计算单核苷酸的百分比
    for region, sequence in sequences.items():
        for nt in single_nts:
            features.append(calculate_base_percent(sequence, 1).get(nt, 0))

    # 计算双核苷酸的百分比
    for region, sequence in sequences.items():
        for dinuc in di_nts:
            features.append(calculate_base_percent(sequence, 2).get(dinuc, 0))

    # 计算三核苷酸的百分比
    for region, sequence in sequences.items():
        for trinuc in tri_nts:
            features.append(calculate_base_percent(sequence, 3).get(trinuc, 0))

    return np.array(features)


# 定义用于计算自由能的函数
def calculate_watson_crick_free_energy(sense_seq, antisense_seq):
    """
    计算 Watson-Crick 配对的自由能，忽略 antisense 序列的最后两个碱基（尾巴）
    :param sense_seq: siRNA 的正义链 (sense)
    :param antisense_seq: siRNA 的反义链 (antisense)
    :return: Watson-Crick 自由能
    """
    # 去掉 antisense_seq 的最后两个碱基
    antisense_seq_trimmed = antisense_seq[:-2]
    
    # 使用 ViennaRNA 的 RNAcofold 模块计算自由能
    cofold = RNA.cofold(sense_seq + "&" + antisense_seq_trimmed)
    
    # 返回配对的自由能值
    return np.array([cofold[1]])


def calculate_end_duplex_energy_difference(sense_seq, antisense_seq):
    """
    计算 siRNA 5' 端和 3' 端前 5 个核苷酸的双链自由能差异
    :param sense_seq: siRNA 的正义链 (sense)
    :param antisense_seq: siRNA 的反义链 (antisense)
    :return: 5' 和 3' 端自由能差异
    """
    antisense_seq_trimmed = antisense_seq[:-2]

    # 提取 5' 端和 3' 端前 5 个核苷酸
    sense_5_prime = sense_seq[:5]
    antisense_5_prime = antisense_seq_trimmed[:5]
    sense_3_prime = sense_seq[-5:]
    antisense_3_prime = antisense_seq_trimmed[-5:]

    # 计算 5' 端的自由能
    five_prime_result = RNA.duplexfold(sense_5_prime, antisense_5_prime)
    five_prime_energy = five_prime_result.energy  # 访问 energy 属性
    
    # 计算 3' 端的自由能
    three_prime_result = RNA.duplexfold(sense_3_prime, antisense_3_prime)
    three_prime_energy = three_prime_result.energy  # 访问 energy 属性
    
    # 返回两者的差值
    return np.array([five_prime_energy - three_prime_energy])



def calculate_interaction_energy(antisense_seq, mrna_seq):
    """
    计算 siRNA 与 mRNA 结合的总自由能
    :param sense_seq: siRNA 的正义链 (sense)
    :param antisense_seq: siRNA 的反义链 (antisense)
    :param mrna_seq: mRNA 靶标序列
    :return: siRNA 和 mRNA 结合的自由能
    """
    antisense_seq_trimmed = antisense_seq[:-2]
    
    # 使用 RNAcofold 计算 siRNA (antisense) 与 mRNA 的配对自由能
    cofold_result = RNA.cofold(antisense_seq_trimmed + "&" + mrna_seq)
    
    # 返回结合自由能
    return np.array([cofold_result[1]])



def plot_training_curves(train_loss_list, val_loss_list, score_list, title, addr):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    fig.suptitle(title, fontsize=16)
    
    ax1.plot(train_loss_list, label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(val_loss_list, label='Validation Loss', color='orange')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    ax3.plot(score_list, label='Score', color='green')
    ax3.set_title('Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()

    save_path = os.path.join(addr,title+'.png')
    
    plt.savefig(save_path)
