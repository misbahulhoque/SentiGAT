import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import copy
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import metrics, preprocessing
import os
import itertools
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GCNConv, GATConv

from helpers_sentiGAT import get_features, object_word_alignment_robust

outloc = '../outputs/'
model_dir = '../saved_models/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Train SentiGAT model')
    parser.add_argument('--mvsa', type=str, default='single', choices=['single', 'multiple'])
    parser.add_argument('--batch-size', type=int, default=64)       # 32, 64, 128
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)      
    parser.add_argument('--split', type=int, default=1)             # from 1 to 10
    #parser.add_argument('--feature-dim', type=int, default=512)    # 512 for CLIP      
    parser.add_argument('--drop-out', type=float, default=0.3)
    parser.add_argument('--hidden-dim', type=int, default=512)      
    
    return parser.parse_args()

def collate_fn(batch):
    sentence_feats, global_feats, imgtxt_feats, face_feats, word_feats, object_feats, labels = zip(*batch)
    
    # Convert fixed-size features to tensors
    sentence_feats = torch.stack([torch.as_tensor(f) for f in sentence_feats])
    global_feats = torch.stack([torch.as_tensor(f) for f in global_feats])
    imgtxt_feats = torch.stack([torch.as_tensor(f) for f in imgtxt_feats])
    face_feats = torch.stack([torch.as_tensor(f) for f in face_feats])
    labels = torch.as_tensor(labels)
    
    # Pad variable-length word and object features
    def pad_features(feature_list, max_len=None):
        if max_len is None:
            max_len = max([f.shape[0] for f in feature_list])
        padded = [torch.cat([f, torch.zeros(max_len - f.shape[0], 512, device=device)], dim=0) for f in feature_list]
        return torch.stack(padded)
    
    # Pad word features
    word_lengths = [f.shape[0] for f in word_feats]
    max_word_len = max(word_lengths)
    padded_word_feats = pad_features(word_feats, max_word_len)
    
    # Pad object features
    object_lengths = [f.shape[0] for f in object_feats]
    max_object_len = max(object_lengths)
    padded_object_feats = pad_features(object_feats, max_object_len)
    
    return (
        sentence_feats.to(device),
        global_feats.to(device),
        imgtxt_feats.to(device),
        face_feats.to(device),
        padded_word_feats.to(device),
        padded_object_feats.to(device),
        labels.to(device)
    )

class GraphData(Dataset):
    """
    generate graph data for each embedding
    """
    def __init__(self, sentence_emb, global_emb, imgtxt_emb, face_emb, word_emb, object_emb, labels):
        self.sentence_emb = sentence_emb
        self.global_emb = global_emb
        self.imgtxt_emb = imgtxt_emb
        self.face_emb = face_emb
        self.word_emb = word_emb
        self.object_emb = object_emb
        self.labels = np.array(labels).astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence_feat = self.sentence_emb[idx]
        global_feat = self.global_emb[idx]
        imgtxt_feat = self.imgtxt_emb[idx]
        face_feat = self.face_emb[idx]
        word_feat = self.word_emb[idx]
        object_feat = self.object_emb[idx]
        label = self.labels[idx]
        return sentence_feat, global_feat, imgtxt_feat, face_feat, word_feat, object_feat, label

def add_feature_indicator(features):
    """
    Adds a binary feature to indicate whether the input features are valid (non-zero).
    """
    valid_indicator = (features.sum(dim=1) != 0).float().unsqueeze(1)  # Shape: (B, 1)
    return torch.cat([features, valid_indicator], dim=1)  # Shape: (B, D + 1)

class ObjectWordGAT(nn.Module):
    """
    Word-object alignment module
    """
    def __init__(self, in_dim=512, out_dim=512, heads=2):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=False)
        
    def forward(self, object_embs, word_embs):
        batch_size = object_embs.shape[0]
        aligned_texts = []
        
        for b in range(batch_size):
            # Extract non-padded elements (assuming padding is zeros)
            obj_mask = (object_embs[b].sum(dim=1) != 0)
            word_mask = (word_embs[b].sum(dim=1) != 0)
            
            obj = object_embs[b][obj_mask]      # [real_objects, 512]
            words = word_embs[b][word_mask]     # [real_words, 512]
            
            if obj.shape[0] == 0 or words.shape[0] == 0:
                aligned_texts.append(torch.zeros(512).to(object_embs.device))
                continue
            
            # Construct bipartite graph edges
            n_objects = obj.shape[0]
            n_words = words.shape[0]
            edge_sources = torch.arange(n_objects).repeat(n_words)
            edge_targets = torch.arange(n_words).repeat_interleave(n_objects) + n_objects
            edge_index = torch.stack([edge_sources, edge_targets], dim=0).to(obj.device)
            
            # Node features
            x = torch.cat([obj, words], dim=0)
            
            # Apply GAT
            x = self.gat(x, edge_index)
            
            # Split and aggregate
            updated_objects = x[:n_objects]
            updated_words = x[n_objects:]
            
            attn_weights = torch.softmax(updated_objects @ updated_words.T, dim=1)
            weighted_words = torch.einsum('ij,jk->ik', attn_weights, updated_words)
            aligned_text = weighted_words.mean(dim=0)
            aligned_texts.append(aligned_text)
        
        return torch.stack(aligned_texts)  # [B, 512]
    
class SentiGAT(nn.Module):
    """
    Apply SentiGAT model
    """
    def __init__(self, feature_dim=512, hidden_dim=512, num_classes=3, drop_out=0.3):
        super(SentiGAT, self).__init__()
        self.feature_dim = feature_dim
        self.edge_index = torch.tensor([                            # Edge index
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
        ], dtype=torch.long).to(device)

        self.object_word_gat = ObjectWordGAT()
        
        # Attention-based interactions
        self.gat1 = GATConv(feature_dim+1, feature_dim, heads=1)    # First GAT layer
        self.gat2 = GATConv(feature_dim, feature_dim, heads=1)      # Second GAT layer

        # Classification
        self.mlp = nn.Sequential(
            nn.Linear(5 * feature_dim, hidden_dim),                 # Combine all 5 features
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, num_classes)                      # Output layer
        )

    def forward(self, text_feat, image_feat, imgtxt_feat, face_feat, word_feat, object_feat):
        aligned_feat = self.object_word_gat(word_feat, object_feat)

        # Add feature indicators (B, D + 1)
        text_feat = add_feature_indicator(text_feat)                
        image_feat = add_feature_indicator(image_feat)              
        imgtxt_feat = add_feature_indicator(imgtxt_feat)            
        face_feat = add_feature_indicator(face_feat)                
        aligned_feat = add_feature_indicator(aligned_feat)      

        # Normalize the input features
        text_feat = F.normalize(text_feat, p=2, dim=1)              # Normalization
        image_feat = F.normalize(image_feat, p=2, dim=1)
        imgtxt_feat = F.normalize(imgtxt_feat, p=2, dim=1)
        face_feat = F.normalize(face_feat, p=2, dim=1)
        aligned_feat = F.normalize(aligned_feat, p=2, dim=1)

        # Create a graph with 5 nodes (text, image, imgtxt, face, word_object)
        batch_size = text_feat.size(0)
        num_nodes = 5                                               # Number of nodes

        # Stack all features
        x = torch.stack([text_feat, image_feat, imgtxt_feat, face_feat, aligned_feat], dim=1)  # (B, 5, D + 1)

        # Repeat edge_index for each example in the batch
        edge_index = self.edge_index.unsqueeze(0).repeat(batch_size, 1, 1)

        # Reshape for GAT input
        x = x.view(-1, self.feature_dim + 1)
        edge_index = edge_index.view(2, -1)

        # Compute edge weights using cosine similarity
        edge_weights = self.compute_edge_weights(x.view(batch_size, 5, -1))  # [B * 20]

        x = self.gat1(x, edge_index, edge_attr=edge_weights)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_weights)

        # Reshape back to (B, 5, D)
        x = x.view(batch_size, num_nodes, -1)  # (B, 5, D)

        # Extract updated features for all nodes
        updated_text_feat = x[:, 0, :]  # (B, D)
        updated_image_feat = x[:, 1, :]  # (B, D)
        updated_imgtxt_feat = x[:, 2, :]  # (B, D)
        updated_face_feat = x[:, 3, :]  # (B, D)
        updated_aligned_feat = x[:, 4, :]  # (B, D)

        # Combine all features
        fused_feature = torch.cat([
            updated_text_feat, updated_image_feat, updated_imgtxt_feat, updated_face_feat, updated_aligned_feat
        ], dim=1)  # (B, 5 * D)

        # Pass through MLP for classification
        logits = self.mlp(fused_feature)  # (B, num_classes)
        return logits

    def compute_edge_weights(self, x):
        """
        Compute cosine similarity-based edge weights for the graph.
        x: (B, 5, D+1) tensor of modality node features.
        Returns: (B * num_edges,) edge weight tensor.
        """
        batch_size, num_nodes, _ = x.shape
        edge_indices = self.edge_index.T.cpu().numpy()  # Shape: [20, 2]

        similarities = []
        for b in range(batch_size):
            sim_matrix = F.cosine_similarity(
                x[b].unsqueeze(1),  # [5, 1, D]
                x[b].unsqueeze(0),  # [1, 5, D]
                dim=-1
            )  # [5, 5]

            sims = [sim_matrix[src, dst] for src, dst in edge_indices]
            similarities.extend(sims)

        return torch.tensor(similarities, device=x.device)

    
    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
    

def train(model, tr_loader, vl_loader, num_epochs, lr=1e-4):
   
    best_model = None
    best_acc = 0.0
    best_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        model.train()
        current_loss = 0.0
        current_corrects = 0
        total = 0.0
        count = 0
        # Iterate over data.
        for sentence_feat, global_feat, imgtxt_feat, face_feat, word_feat, object_feat, labels in tr_loader:    
            sentence_feat = sentence_feat.to(device).float()
            global_feat = global_feat.to(device).float()
            imgtxt_feat = imgtxt_feat.to(device).float()
            face_feat = face_feat.to(device).float()
            word_feat = word_feat.to(device).float()
            object_feat = object_feat.to(device).float()
            labels = labels.to(device).long() 

            optimizer.zero_grad()

            outputs = model(sentence_feat, global_feat, imgtxt_feat, face_feat, word_feat, object_feat)
            loss = model.compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            current_loss += loss.item()
            current_corrects += torch.sum(preds == labels.data).item()
            total += len(labels)

            if count % 50 == 0:
                print('[%d, %5d] loss: %.5f, Acc: %.2f' %
                    (epoch, count + 1, loss.item(), (100.0 * current_corrects) / total))

            count = count + 1

        train_loss = current_loss / len(tr_loader)
        train_acc = current_corrects * 1.0 / len(tr_loader.dataset)

        print('Training Loss: {:.6f} Acc: {:.2f}'.format(train_loss, 100.0 * train_acc))

        test_loss, test_acc, test_f1 = evaluate(model, vl_loader)

        print('Epoch: {:d}, Val Loss: {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}'.format(epoch, test_loss, test_acc, test_f1))
       
        if test_acc > best_acc:  # Use accuracy instead of loss
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        
        scheduler.step()

    print(f"Best Epoch: {best_epoch} with Val Acc: {best_acc:.4f}")
    return best_model, best_epoch

def evaluate(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sentence_feat, global_feat, imgtxt_feat, face_feat, word_feat, object_feat, labels in loader:
            sentence_feat = sentence_feat.to(device).float()
            global_feat = global_feat.to(device).float()
            imgtxt_feat = imgtxt_feat.to(device).float()
            face_feat = face_feat.to(device).float()
            word_feat = word_feat.to(device).float()
            object_feat = object_feat.to(device).float()
            labels = labels.to(device).long() 

            outputs = model(sentence_feat, global_feat, imgtxt_feat, face_feat, word_feat, object_feat)

            # Compute losses
            loss = model.compute_loss(outputs, labels)
            test_loss += loss.item()
            preds = torch.argmax(outputs.data, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        acc = metrics.accuracy_score(all_labels, all_preds)
        f1 = metrics.f1_score(all_labels, all_preds, average='weighted')

    return test_loss/len(loader), acc, f1

def run(mvsa='single', batch_size=32, init_lr=2e-5, epochs=5, split=1, feature_dim= 512, drop_out= 0.3, hidden_dim=512):
    if mvsa == 'single':
        dloc = '../data/mvsa_single/splits/'
    else:
        dloc = '../data/mvsa_multiple/splits/'

    # Load splits
    tr_ids = pd.read_csv(f'{dloc}train_{split}.txt', header=None).to_numpy().flatten()
    vl_ids = pd.read_csv(f'{dloc}val_{split}.txt', header=None).to_numpy().flatten()
    te_ids = pd.read_csv(f'{dloc}test_{split}.txt', header=None).to_numpy().flatten()

    pair_df = pd.read_csv(f'{dloc}valid_pairlist.txt', header=None)

    all_labels = pair_df[1].to_numpy().flatten()
    

   # Load features
    sentence_emb, word_emb, global_emb, face_emb, object_emb, imgtxt_emb = get_features(mvsa)

    # Convert into tensors
    object_emb = [torch.tensor(obj) if not isinstance(obj, torch.Tensor) else obj for obj in object_emb]
    word_emb = [torch.tensor(wrd) if not isinstance(wrd, torch.Tensor) else wrd for wrd in word_emb]


    tr_data = GraphData(sentence_emb[tr_ids], global_emb[tr_ids], imgtxt_emb[tr_ids], face_emb[tr_ids], [word_emb[i] for i in tr_ids], [object_emb[i] for i in tr_ids], all_labels[tr_ids])
    vl_data = GraphData(sentence_emb[vl_ids], global_emb[vl_ids], imgtxt_emb[vl_ids], face_emb[vl_ids], [word_emb[i] for i in vl_ids], [object_emb[i] for i in vl_ids], all_labels[vl_ids])
    te_data = GraphData(sentence_emb[te_ids], global_emb[te_ids], imgtxt_emb[te_ids], face_emb[te_ids], [word_emb[i] for i in te_ids], [object_emb[i] for i in te_ids], all_labels[te_ids])


    if __name__ == '__main__':
        tr_loader = DataLoader(dataset=tr_data, batch_size=batch_size, num_workers=0,
                               shuffle=True, collate_fn=collate_fn, )
        vl_loader = DataLoader(dataset=vl_data, batch_size=16, num_workers=0, collate_fn=collate_fn, )
        te_loader = DataLoader(dataset=te_data, batch_size=16, num_workers=0, collate_fn=collate_fn, )


        model_SG = SentiGAT(
            feature_dim = 512, 
            hidden_dim = hidden_dim,
            num_classes = 3,
            drop_out = drop_out,
        ).to(device)
        print(model_SG)

        model_SG, best_epoch = train(model_SG, tr_loader, vl_loader, num_epochs=epochs, lr=init_lr)
        
        #torch.save(model_SG.state_dict(), os.path.join(model_dir, f'best_model_{mvsa}_{split}.pth'))
        #print(f"model saved")
        
        te_loss, te_acc, te_f1 = evaluate(model_SG, te_loader)
        print(f'Best Epoch: {best_epoch}, Test Acc: {np.round(te_loss, 4)}, {np.round(te_acc, 4)}, {np.round(te_f1, 4)}')

        result_file = f'{outloc}/SentiGAT_{mvsa}.csv'

        result_data = {
            'mvsa': [mvsa],
            'split': [split],
            'epochs': [epochs],
            'batch_size': [batch_size],
            'init_lr': [init_lr],
            'hidden_dim': [hidden_dim], 
            'drop_out': [drop_out], 
            'te_loss': [np.round(te_loss, 4)],
            'te_acc': [np.round(te_acc, 4)],
            'te_f1': [np.round(te_f1, 4)]
        }
        result_df = pd.DataFrame(result_data)

        # Append to CSV file or create a new one
        result_df.to_csv(result_file, mode='a', header=not os.path.exists(result_file), index=False)

        print(f'epochs: {epochs} bs: {batch_size} lr: {init_lr} split: {split}')
        print('-' * 30)
        print('DONE')
    


def main():
    args = parse_args()
    run(mvsa=args.mvsa, batch_size=args.batch_size, init_lr=args.lr, epochs=args.epochs,
         split=args.split, feature_dim= 512, drop_out=args.drop_out, hidden_dim=args.hidden_dim)

if __name__ == '__main__':
    main()
