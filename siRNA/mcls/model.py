import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
     
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return self.relu(out)


class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleCNN, self).__init__()
        # 第一组卷积
        self.conv1_1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        
        # 第二组卷积
        self.conv2_1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        
        # 第三组卷积
        self.conv3_1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.conv3_2 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        
        # 第四组卷积
        self.conv4_1 = nn.Conv1d(in_channels, 32, kernel_size=13, padding=6)
        self.conv4_2 = nn.Conv1d(32, 32, kernel_size=13, padding=6)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1_1(x))
        out1 = self.relu(self.conv1_2(out1))
        
        out2 = self.relu(self.conv2_1(x))
        out2 = self.relu(self.conv2_2(out2))
        
        out3 = self.relu(self.conv3_1(x))
        out3 = self.relu(self.conv3_2(out3))
        
        out4 = self.relu(self.conv4_1(x))
        out4 = self.relu(self.conv4_2(out4))
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class SiRNAModel(nn.Module):
    def __init__(self, vocab_size, cat_mapping_len, embed_dim=200, cat_embed_dim=32, hidden_dim=256, n_layers=3, dropout=0.5, num_dim=2, device='cuda'):
        super(SiRNAModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.cat_embedding = nn.ModuleList([nn.Embedding(num, cat_embed_dim) for num in cat_mapping_len])
        
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        
        self.seq_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        # Multi-scale CNN with residual connections
        self.convseq = nn.Sequential(
            ResidualBlock(in_channels=200, out_channels=200, kernel_size=3, padding=1),
            MultiScaleCNN(in_channels=200),
            nn.Conv1d(in_channels= 128 , out_channels= 256, kernel_size=3, padding=1),##应该增加通道数
            nn.AdaptiveAvgPool1d(1)
        )

        self.cat_net = nn.Sequential(
            nn.Linear(len(cat_mapping_len) * cat_embed_dim, hidden_dim*2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.num_net = nn.Sequential(
            nn.Linear(num_dim, hidden_dim*2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        
        self.fc = nn.Sequential(
            nn.Linear(2061, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    

    def get_seq_feat(self,seqs):
        # Use CNN and RNN to model sequence features ()
        embedded = [self.embedding(seq) for seq in seqs]
        rnn_outputs = []
        cnn_outputs = []

        # RNN process
        for embed in embedded:
            x, _ = self.rnn(embed)
            x = self.dropout(x[:, -1, :])
            rnn_outputs.append(x)


        # CNN process
        for embed in embedded:
            x = self.convseq(embed.permute(0, 2, 1))
            x = torch.flatten(x, start_dim=1)
            cnn_outputs.append(x)

        antisense_feat = self.seq_fc(torch.cat((rnn_outputs[0],cnn_outputs[0]),dim=1))
        sense_feat = self.seq_fc(torch.cat((rnn_outputs[1],cnn_outputs[1]),dim=1))
        modified_antisense_feat = self.seq_fc(torch.cat((rnn_outputs[2],cnn_outputs[2]),dim=1))
        modified_sense_feat = self.seq_fc(torch.cat((rnn_outputs[3],cnn_outputs[3]),dim=1))
        mrna_feat =self.seq_fc(torch.cat((rnn_outputs[4],cnn_outputs[4]),dim=1))

        seq_feat = {
            'antisense_feat' : antisense_feat,
            'sense_feat': sense_feat,
            'modified_antisense_feat': modified_antisense_feat,
            'modified_sense_feat':modified_sense_feat,
            'mrna_feat': mrna_feat
        }

        return seq_feat
    
    def get_predition(self,seq_feat,cat_data,num_data,prior):

        cat_data = torch.cat(cat_data, dim=0).to(dtype=torch.long)
        cat_data = [embed(cat_data[:, i]) for i, embed in enumerate(self.cat_embedding)]
        cat_data = torch.cat(cat_data, dim=1)
        cat_feat = self.cat_net(cat_data)


        antisense_feat = seq_feat['antisense_feat']
        sense_feat = seq_feat['sense_feat']
        modified_sense_feat = seq_feat['modified_sense_feat']
        mrna_feat = seq_feat['mrna_feat']
        modified_antisense_feat = seq_feat['modified_antisense_feat']


        num_feat = self.num_net(num_data)
        x = self.fc(torch.cat((antisense_feat, sense_feat,modified_sense_feat,modified_antisense_feat,mrna_feat, cat_feat, num_feat,prior.float()),dim=-1))

        return x.squeeze()

    def forward(self, seqs, cat_data, num_data,prior):
        seq_feat= self.get_seq_feat(seqs) 
        prediction = self.get_predition(seq_feat,cat_data,num_data,prior)
        return seq_feat,prediction



