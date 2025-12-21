"""
Multi-Hold-Out OOD Detection Experiment

Evaluates BSTGCN's OOD detection across multiple attack types as hold-out classes:
- SlowrateDoS (low-rate attack)
- SYNScan (scan-type attack)
- ICMPFlood (flood-type attack)

This provides more robust evaluation of uncertainty estimation generalization.
"""

import os
import sys
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent if '__file__' in dir() else Path('.')
sys.path.insert(0, str(project_root))

from scripts.BSTGCN.bstgcn_model import BSTGCNModel as BSTGCN
from scripts.BSTGCN.mc_inference import mc_inference

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

CLASS_NAMES = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan',
               'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan']

# Hold-out classes for OOD evaluation (diverse attack types)
HOLDOUT_CLASSES = {
    'SlowrateDoS': 5,    # Low-rate attack
    'SYNScan': 4,        # Scan-type attack
    'ICMPFlood': 2,      # Flood-type attack
}


def prepare_ood_data(data_dir, ood_label, batch_size=128):
    """Prepare data with one class held out as OOD"""
    train_seq = np.load(os.path.join(data_dir, 'train_sequences.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    test_seq = np.load(os.path.join(data_dir, 'test_sequences.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))

    # Separate OOD samples
    train_id_mask = train_labels != ood_label
    test_id_mask = test_labels != ood_label
    test_ood_mask = test_labels == ood_label

    # Create label mapping (remove OOD class)
    unique_labels = sorted([l for l in np.unique(train_labels) if l != ood_label])
    label_map = {old: new for new, old in enumerate(unique_labels)}

    train_id_seq = train_seq[train_id_mask]
    train_id_labels = np.array([label_map[l] for l in train_labels[train_id_mask]])
    test_id_seq = test_seq[test_id_mask]
    test_id_labels = np.array([label_map[l] for l in test_labels[test_id_mask]])
    test_ood_seq = test_seq[test_ood_mask]

    # Split train into train/val
    train_seq_final, val_seq, train_labels_final, val_labels = train_test_split(
        train_id_seq, train_id_labels, test_size=0.15, stratify=train_id_labels, random_state=42
    )

    # Normalize
    n_train, seq_len, n_feat = train_seq_final.shape
    scaler = StandardScaler()
    train_flat = train_seq_final.reshape(-1, n_feat)
    scaler.fit(train_flat)

    train_seq_final = scaler.transform(train_flat).reshape(n_train, seq_len, n_feat)
    val_seq = scaler.transform(val_seq.reshape(-1, n_feat)).reshape(-1, seq_len, n_feat)
    test_id_seq = scaler.transform(test_id_seq.reshape(-1, n_feat)).reshape(-1, seq_len, n_feat)
    test_ood_seq = scaler.transform(test_ood_seq.reshape(-1, n_feat)).reshape(-1, seq_len, n_feat)

    # Create loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_seq_final), torch.LongTensor(train_labels_final)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_seq), torch.LongTensor(val_labels)),
        batch_size=batch_size, shuffle=False
    )
    test_id_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_id_seq), torch.LongTensor(test_id_labels)),
        batch_size=batch_size, shuffle=False
    )
    test_ood_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_ood_seq), torch.zeros(len(test_ood_seq), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    metadata = {
        'num_classes': len(unique_labels),
        'num_features': n_feat,
        'seq_len': seq_len,
        'train_samples': len(train_seq_final),
        'val_samples': len(val_seq),
        'test_id_samples': len(test_id_seq),
        'test_ood_samples': len(test_ood_seq),
    }

    return train_loader, val_loader, test_id_loader, test_ood_loader, metadata


def train_and_evaluate(train_loader, val_loader, test_id_loader, test_ood_loader,
                       metadata, config, device):
    """Train BSTGCN and evaluate OOD detection"""
    model = BSTGCN(
        input_dim=metadata['num_features'],
        hidden_dim=config['hidden_dim'],
        num_classes=metadata['num_classes'],
        num_st_blocks=config['num_st_blocks'],
        k_neighbors=config['k_neighbors'],
        prior_std=config['prior_std'],
        dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    best_val_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Training
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, sample=True)
            kl_loss = model.kl_divergence()
            ce_loss = criterion(logits, batch_y)
            kl_weight = min(1.0, epoch / config['kl_warmup_epochs']) * config['kl_weight']
            loss = ce_loss + kl_weight * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x, sample=False)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.numpy())
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, best_val_f1


def evaluate_ood(model, test_id_loader, test_ood_loader, device, num_samples=30):
    """Evaluate OOD detection using epistemic uncertainty"""
    model.eval()

    id_uncertainties, id_preds, id_labels = [], [], []
    for batch_x, batch_y in test_id_loader:
        batch_x = batch_x.to(device)
        _, pred_class, epistemic = mc_inference(model, batch_x, num_samples=num_samples)
        id_uncertainties.extend(epistemic.cpu().numpy())
        id_preds.extend(pred_class.cpu().numpy())
        id_labels.extend(batch_y.numpy())

    ood_uncertainties = []
    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, _, epistemic = mc_inference(model, batch_x, num_samples=num_samples)
        ood_uncertainties.extend(epistemic.cpu().numpy())

    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    # OOD detection metrics (using uncertainty as OOD score)
    y_true = np.concatenate([np.zeros(len(id_uncertainties)), np.ones(len(ood_uncertainties))])
    y_scores = np.concatenate([id_uncertainties, ood_uncertainties])

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    return {'auroc': auroc, 'aupr': aupr, 'id_acc': id_acc, 'id_f1': id_f1}


def main():
    parser = argparse.ArgumentParser(description='Multi-Hold-Out OOD Detection')
    parser.add_argument('--data-dir', type=str, default='data/processed/raw_packet_sequences_50packets_23feat')
    parser.add_argument('--save-dir', type=str, default='results/multi_holdout_ood')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = {
        'hidden_dim': 128, 'num_st_blocks': 3, 'k_neighbors': 10,
        'prior_std': 1.5, 'dropout': 0.15, 'lr': 8e-4, 'weight_decay': 1e-5,
        'kl_weight': 2e-5, 'kl_warmup_epochs': 40, 'label_smoothing': 0.05,
        'epochs': args.epochs, 'patience': args.patience
    }

    results = {}

    for ood_name, ood_label in HOLDOUT_CLASSES.items():
        print(f"\n{'='*60}")
        print(f"Hold-Out Class: {ood_name} (label={ood_label})")
        print(f"{'='*60}")

        train_loader, val_loader, test_id_loader, test_ood_loader, metadata = \
            prepare_ood_data(args.data_dir, ood_label)

        print(f"  ID Classes: {metadata['num_classes']}")
        print(f"  Train: {metadata['train_samples']}, Val: {metadata['val_samples']}")
        print(f"  ID Test: {metadata['test_id_samples']}, OOD Test: {metadata['test_ood_samples']}")

        model, val_f1 = train_and_evaluate(
            train_loader, val_loader, test_id_loader, test_ood_loader,
            metadata, config, device
        )
        print(f"  Best Val F1: {val_f1:.4f}")

        ood_results = evaluate_ood(model, test_id_loader, test_ood_loader, device)
        results[ood_name] = ood_results
        print(f"  ID Acc: {ood_results['id_acc']:.4f}, ID F1: {ood_results['id_f1']:.4f}")
        print(f"  AUROC: {ood_results['auroc']:.4f}, AUPR: {ood_results['aupr']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("Multi-Hold-Out OOD Detection Summary")
    print(f"{'='*60}")
    print(f"{'Hold-Out Class':<15} {'AUROC':<10} {'AUPR':<10} {'ID Acc':<10} {'ID F1':<10}")
    print("-" * 55)
    aurocs, auprs = [], []
    for name, res in results.items():
        print(f"{name:<15} {res['auroc']:.4f}     {res['aupr']:.4f}     {res['id_acc']:.4f}     {res['id_f1']:.4f}")
        aurocs.append(res['auroc'])
        auprs.append(res['aupr'])
    print("-" * 55)
    print(f"{'Average':<15} {np.mean(aurocs):.4f}     {np.mean(auprs):.4f}")
    print(f"{'Std':<15} {np.std(aurocs):.4f}     {np.std(auprs):.4f}")

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.save_dir, f'multi_holdout_results_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()

