import torch
from torch.utils.data import Dataset
import numpy as np

class SiRNADataset(Dataset):
    def __init__(self, df, seq_columns, cat_columns, num_columns,prior_columns, vocab, tokenizer, max_len_sirna):
        self.df = df
        self.seq_columns = seq_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.prior_columns = prior_columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len_sirna = max_len_sirna

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seqs = [self.tokenize_and_encode(row[col]) for col in self.seq_columns]
        cat_data = torch.tensor([row[col] for col in self.cat_columns]).unsqueeze(0)
        num_data = torch.tensor([row[col] for col in self.num_columns], dtype=torch.float)#.unsqueeze(0)  pay attention to dimension
        prior = torch.tensor(np.concatenate([row[col] for col in self.prior_columns]))
        target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)

        return seqs, cat_data, num_data,prior, target

    def tokenize_and_encode(self, seq):
        if ' ' in seq:  # Modified sequence
            tokens = seq.split()
        else:  # Regular sequence
            tokens = self.tokenizer.tokenize(seq)
        
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]  # Use 0 (pad) for unknown tokens
        max_len = self.max_len_sirna
        
        padded = encoded + [0] * (max_len - len(encoded))
        return torch.tensor(padded[:max_len], dtype=torch.long)
