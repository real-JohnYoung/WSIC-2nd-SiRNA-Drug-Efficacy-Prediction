import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rich import print
import argparse
import random
from model import  SiRNAModel
from utils import calculate_metrics,set_seed,get_priors,CustomLoss,get_priors_,GenomicTokenizer,GenomicVocab,find_best_antisense_match,apply_parallel,plot_training_curves,evaluate_model
from dataset import SiRNADataset
import pickle
import json

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=51, device='cuda'):
    model.to(device)
    best_score = -float('inf')
    best_model = None
    criterion =criterion.to(device)

    train_loss_list = []
    val_loss_list = []
    score_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for seqs, cat_data, num_data, prior,targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            seqs = [x.to(device) for x in seqs]
            cat_data = [x.to(device) for x in cat_data]
            num_data = num_data.to(device)
            prior = prior.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            seq_feat, prediction = model(seqs, cat_data, num_data,prior)
            loss = criterion(seq_feat, prediction, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for seqs, cat_data, num_data, prior,targets in val_loader:
                seqs = [x.to(device) for x in seqs]
                cat_data = [x.to(device) for x in cat_data]
                num_data = num_data.to(device)
                prior = prior.to(device)
                targets = targets.to(device)
                seq_feat,prediction = model(seqs, cat_data, num_data,prior)
                loss = criterion(seq_feat,prediction,targets)
                val_loss += loss.item()
                val_preds.extend(prediction.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_preds = np.clip(val_preds,0,100)
        val_targets = np.clip(val_targets,0,100)
        score = calculate_metrics(val_targets, val_preds)
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        score_list.append(score)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Validation Score: {score:.4f}')

        if score > best_score:
            best_score = score
            best_model = model
            print(f'New best model found with score: {best_score:.4f}')

    print("best score", best_score)
    torch.save(best_model.state_dict(), f'best_model.pth')
    
    return model, train_loss_list, val_loss_list, score_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train MCLS Model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and inference")
    parser.add_argument("--epoch", type=int, default=60, help="Epoch for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Epoch for training")
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()

    device = args.device
    # Load data
    train_data = pd.read_csv('../dataset/train_data.csv')


    seq_columns = ['siRNA_antisense_seq','siRNA_sense_seq','modified_siRNA_antisense_seq_list', 'modified_siRNA_sense_seq_list','Extended_Sequence']
    cat_columns = ['cell_line_donor', 'Transfection_method', 'gene_target_species', 'gene_target_ncbi_id']
    num_columns = ['siRNA_concentration', 'Duration_after_transfection_h','Match_Score','Seed_Match_Score']
    prior_columns = [ 'gc_sterch', 'gc_features', 'gibbs_energy_result', 'single_nt_percent', 'di_nt_percent', 'tri_nt_percent', 'secondary_struct',
                      'mrna_secondary_struct', 'mrna_features','watson_crick_free_energy','interaction_energy','check_bases','encode_modifications']
    data_columns = cat_columns + num_columns
    
    train_data.dropna(subset=['siRNA_antisense_seq','gene_target_seq'], inplace=True)

    train_results = apply_parallel(train_data, find_best_antisense_match)
    if train_results:
        processed_sequences, matched_sequences, match_scores, extended_sequences , seed_match_scores = zip(*train_results)
    else:
        processed_sequences, matched_sequences, match_scores, extended_sequences, seed_match_scores = None, None, None, None, None
        
    train_data['processed_siRNA_antisense_seq'] = processed_sequences
    train_data['Matched_Sequence'] = matched_sequences
    train_data['Match_Score'] = match_scores
    train_data['Extended_Sequence'] = extended_sequences
    train_data['Seed_Match_Score'] = seed_match_scores

    train_data.dropna(subset=data_columns + seq_columns + ['mRNA_remaining_pct'], inplace=True)


    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=args.random_state)
    train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=args.random_state)
    
    set_seed(args.random_state)

    cat_mapping_set = []
    cat_mapping_len = []

    for col in cat_columns:
        train_data[col], unique = pd.factorize(train_data[col])
        mapping = {value: code for code, value in enumerate(unique)}
        cat_mapping_len.append(len(mapping))
        cat_mapping_set.append(mapping)
    
    with open('cat_mapping_set.pkl', 'wb') as f:
        pickle.dump(cat_mapping_set, f)

    with open('cat_mapping_len.pkl', 'wb') as f:
        pickle.dump(cat_mapping_len, f)

    train_data = get_priors(train_data)
    val_data = get_priors(val_data)
    test_data = get_priors(val_data)


    for i, col in enumerate(cat_columns):
        val_data[col] = val_data[col].map(cat_mapping_set[i])
        test_data[col]= test_data[col].map(cat_mapping_set[i])


    tokenizer = GenomicTokenizer(ngram=3, stride=3)

    all_tokens = []
    for col in seq_columns:
        for seq in train_data[col]:
            if ' ' in seq:  # Modified sequence
                all_tokens.extend(seq.split())
            else:
                all_tokens.extend(tokenizer.tokenize(seq))
    vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
        

    max_len_sirna = max(max(len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq))
                            for seq in train_data[col]) for col in ['siRNA_antisense_seq','siRNA_sense_seq','modified_siRNA_antisense_seq_list', 'modified_siRNA_sense_seq_list'])

    train_dataset = SiRNADataset(train_data, seq_columns, cat_columns, num_columns, prior_columns, vocab, tokenizer, max_len_sirna)
    val_dataset = SiRNADataset(val_data, seq_columns, cat_columns, num_columns, prior_columns, vocab, tokenizer, max_len_sirna)
    test_dataset = SiRNADataset(test_data, seq_columns, cat_columns, num_columns, prior_columns, vocab, tokenizer, max_len_sirna)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = SiRNAModel(len(vocab.itos), cat_mapping_len, num_dim=len(num_columns),device= device)
    criterion = CustomLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()),lr = args.lr, weight_decay=1e-2, eps=1e-8)

    model, train_loss_list,val_loss_list,score_list = train_model(model, train_loader, val_loader, criterion, optimizer, args.epoch, device)

    save_addr = '/data/aidd-1/yc23/WSIC-2nd-SiRNA-Drug-Efficacy-Prediction/siRNA/result_plot'
    title = 'MCLS method'
    plot_training_curves(train_loss_list, val_loss_list, score_list, title, saaddr)
    print(train_loss_list,val_loss_list,score_list)
    evaluate_model(model, test_loader, device=device)
