"""快速评估已训练的BSTGCN模型"""

import os
import sys
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

from scripts.BSTGCN.bstgcn_model import BSTGCNModel
from scripts.BSTGCN.train_bstgcn import evaluate_deterministic, evaluate_bayesian, load_data

from torch.utils.data import DataLoader, TensorDataset

def main():
    # Load data
    data_dir = Path('data/processed/raw_packet_sequences_50packets_23feat')
    train_seq, train_labels, val_seq, val_labels, test_seq, test_labels_arr, metadata = load_data(data_dir)

    # Create test dataloader
    test_dataset = TensorDataset(
        torch.FloatTensor(test_seq),
        torch.LongTensor(test_labels_arr)
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Initialize model
    model = BSTGCNModel(
        input_dim=23,
        hidden_dim=128,
        num_heads=8,
        num_st_blocks=3,
        num_classes=9,
        dropout=0.15,
        k_neighbors=10,
        prior_std=1.0
    )

    # Find the latest model
    result_dirs = list(Path('baseline/results/bstgcn_50packets_23feat').glob('20*'))
    if not result_dirs:
        print("No trained model found!")
        return

    latest_dir = max(result_dirs, key=lambda x: x.name)
    model_path = latest_dir / 'best_model.pth'
    print(f'Loading model from: {model_path}')

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Best Epoch: {checkpoint["epoch"]}, Val F1: {checkpoint["val_f1"]:.4f}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Deterministic evaluation
    print('\n' + '='*60)
    print('Deterministic Evaluation (MAP)')
    print('='*60)
    test_acc_det, test_f1_det, test_preds_det, test_labels = evaluate_deterministic(
        model, test_loader, device
    )
    print(f'Accuracy: {test_acc_det:.4f}')
    print(f'Macro F1: {test_f1_det:.4f}')

    # Class names
    class_names = metadata.get('class_names', [str(i) for i in range(9)])
    print('\nClassification Report (Deterministic):')
    print(classification_report(test_labels, test_preds_det, target_names=class_names, digits=4))

    # MC evaluation
    print('\n' + '='*60)
    print('Bayesian Evaluation (MC Inference, 50 samples)')
    print('='*60)
    test_acc_mc, test_f1_mc, test_unc, test_preds_mc, test_uncertainties, _, _ = evaluate_bayesian(
        model, test_loader, device, num_mc_samples=50
    )
    print(f'Accuracy: {test_acc_mc:.4f}')
    print(f'Macro F1: {test_f1_mc:.4f}')
    print(f'Mean Uncertainty: {test_unc:.4f}')

    # Uncertainty-error correlation
    errors_det = (np.array(test_labels) != np.array(test_preds_det)).astype(float)
    unc_err_corr = np.corrcoef(errors_det, test_uncertainties)[0, 1]
    print(f'\nUncertainty-Error Correlation: {unc_err_corr:.4f}')

    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'BSTGCN (Deterministic): Accuracy={test_acc_det:.4f}, Macro F1={test_f1_det:.4f}')
    print(f'BSTGCN (MC Inference):  Accuracy={test_acc_mc:.4f}, Macro F1={test_f1_mc:.4f}')
    print(f'STGCN Reference:        Accuracy=0.9710, Macro F1=0.9306')

if __name__ == '__main__':
    main()

