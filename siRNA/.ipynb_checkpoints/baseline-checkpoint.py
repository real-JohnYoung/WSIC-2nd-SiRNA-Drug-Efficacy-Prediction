import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from rich import print
from sklearn.metrics import precision_score, recall_score, mean_absolute_error


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


class SiRNADataset(Dataset):
    def __init__(self, df, columns, vocab, tokenizer, max_len):
        self.df = df
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seqs = [self.tokenize_and_encode(row[col]) for col in self.columns]
        target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)

        return seqs, target

    def tokenize_and_encode(self, seq):
        if ' ' in seq:  # Modified sequence
            tokens = seq.split()
        else:  # Regular sequence
            tokens = self.tokenizer.tokenize(seq)
        
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]  # Use 0 (pad) for unknown tokens
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded[:self.max_len], dtype=torch.long)


class SiRNAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=256, n_layers=3, dropout=0.5):
        super(SiRNAModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 4, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = [self.embedding(seq) for seq in x]
        outputs = []
        for embed in embedded:
            x, _ = self.gru(embed)
            x = self.dropout(x[:, -1, :])  # Use last hidden state
            outputs.append(x)
        
        x = torch.cat(outputs, dim=1)
        x = self.fc(x)
        return x.squeeze()


def calculate_metrics(y_true, y_pred, threshold=30):
    mae = np.mean(np.abs(y_true - y_pred))

    y_true_binary = (y_true < threshold).astype(int)
    y_pred_binary = (y_pred < threshold).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100

    precision = precision_score(y_true_binary, y_pred_binary, average='binary')
    recall = recall_score(y_true_binary, y_pred_binary, average='binary')
    f1 = 2 * precision * recall / (precision + recall)
    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return score


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    best_score = -float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        score = calculate_metrics(val_targets, val_preds)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Validation Score: {score:.4f}')

        if score > best_score:
            best_score = score
            best_model = model.state_dict().copy()
            print(f'New best model found with socre: {best_score:.4f}')

    return best_model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.numpy())

    y_pred = np.array(predictions)
    y_test = np.array(targets)
    
    score = calculate_metrics(y_test, y_pred)
    print(f"Test Score: {score:.4f}")



if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    train_data = pd.read_csv('train_data.csv')

    columns = ['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list']
    train_data.dropna(subset=columns + ['mRNA_remaining_pct'], inplace=True)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create vocabulary
    tokenizer = GenomicTokenizer(ngram=3, stride=3)

    all_tokens = []
    for col in columns:
        for seq in train_data[col]:
            if ' ' in seq:  # Modified sequence
                all_tokens.extend(seq.split())
            else:
                all_tokens.extend(tokenizer.tokenize(seq))
    vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)

    # Find max sequence length
    max_len = max(max(len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq)) 
                      for seq in train_data[col]) for col in columns)
    # Create datasets
    train_dataset = SiRNADataset(train_data, columns, vocab, tokenizer, max_len)
    val_dataset = SiRNADataset(val_data, columns, vocab, tokenizer, max_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    model = SiRNAModel(len(vocab.itos))
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())

    train_model(model, train_loader, val_loader, criterion, optimizer, 50, device)
