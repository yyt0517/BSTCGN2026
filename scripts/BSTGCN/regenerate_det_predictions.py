# -*- coding: utf-8 -*-
import numpy as np
import torch
import json
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.BSTGCN.bstgcn_model import BSTGCNModel
from torch.utils.data import DataLoader, TensorDataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    results_dir = os.path.join(project_root, 'baseline/results/bstgcn_50packets_23feat/20251220_152732')
    data_dir = os.path.join(project_root, 'data/processed/raw_packet_sequences_50packets_23feat')

    print('Loading data...')
    test_x = np.load(os.path.join(data_dir, 'test_sequences.npy'))
    test_y = np.load(os.path.join(data_dir, 'test_labels.npy'))
    print(f'  Test samples: {len(test_y)}')
    
    num_classes = len(np.unique(test_y))
    num_features = test_x.shape[2]
    print(f'  Num classes: {num_classes}, Num features: {num_features}')

    with open(os.path.join(results_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    print('Creating model...')
    model = BSTGCNModel(
        input_dim=num_features,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_st_blocks=config['num_st_blocks'],
        num_classes=num_classes,
        dropout=config['dropout'],
        k_neighbors=config['k_neighbors'],
        sage_aggr=config['sage_aggr'],
        prior_std=config['prior_std']
    )

    # Load checkpoint (not just state_dict)
    checkpoint = torch.load(os.path.join(results_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'  Model loaded with {model.get_num_params()} parameters')

    print('Running deterministic inference...')
    test_tensor = torch.FloatTensor(test_x)
    test_loader = DataLoader(TensorDataset(test_tensor, torch.LongTensor(test_y)), batch_size=512)

    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x, sample=False)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())

    all_preds = np.array(all_preds)
    np.save(os.path.join(results_dir, 'test_predictions.npy'), all_preds)
    print(f'Deterministic predictions saved: {all_preds.shape}')

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(test_y, all_preds)
    f1 = f1_score(test_y, all_preds, average='macro')
    print(f'Results: Accuracy={acc:.4f}, Macro F1={f1:.4f}')

if __name__ == '__main__':
    main()
