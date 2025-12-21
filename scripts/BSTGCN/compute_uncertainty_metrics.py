import numpy as np
from pathlib import Path


def compute_auroc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC for binary labels using rank-based formula.

    labels: 1 for positive (error), 0 for negative (correct)
    scores: higher score => more likely positive (higher uncertainty)
    """
    labels = np.asarray(labels).astype(float)
    scores = np.asarray(scores).astype(float)

    n = labels.shape[0]
    if n == 0:
        return float("nan")

    n_pos = labels.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank scores in ascending order (low score -> low rank)
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Sum of ranks of positive samples
    sum_ranks_pos = ranks[labels == 1].sum()

    # Mann–Whitney U / Wilcoxon rank-sum relation
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    base = Path("baseline\results\bstgcn_50packets\20251121_112445")
    data_dir = Path("data/processed/raw_packet_sequences_50packets")

    y_pred = np.load(base / "test_predictions.npy")
    unc = np.load(base / "test_uncertainties.npy")
    y_true = np.load(data_dir / "test_labels.npy")

    if not (y_pred.shape == unc.shape == y_true.shape):
        raise RuntimeError(
            f"Shape mismatch: pred {y_pred.shape}, unc {unc.shape}, y_true {y_true.shape}"
        )

    is_error = (y_pred != y_true).astype(np.int32)
    num_samples = int(is_error.shape[0])
    num_errors = int(is_error.sum())

    print("=== Basic stats ===")
    print(f"Total samples: {num_samples}")
    print(f"Num errors  : {num_errors} ({num_errors / num_samples:.4f})")

    # AUROC: using uncertainty as score to predict error (1 = error)
    auroc = compute_auroc_binary(is_error, unc)
    print("\n=== AUROC (uncertainty -> error) ===")
    print(f"AUROC: {auroc:.4f}")

    # Coverage-Accuracy curve based on uncertainty thresholds
    thresholds = np.quantile(unc, np.linspace(0.0, 1.0, 11))
    print("\n=== Coverage-Accuracy by uncertainty threshold (keep if unc <= t) ===")
    print("idx\tthres\tcoverage\taccuracy\tn_kept")

    for idx, t in enumerate(thresholds):
        keep = unc <= t
        n_kept = int(keep.sum())
        if n_kept == 0:
            continue
        cov = n_kept / num_samples
        acc = float((y_pred[keep] == y_true[keep]).mean())
        print(f"{idx:2d}\t{t:.4f}\t{cov:.4f}\t{acc:.4f}\t{n_kept}")

    # Also report accuracy when keeping lowest-uncertainty X% of samples
    coverages = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]
    sorted_idx = np.argsort(unc)  # ascending: lowest uncertainty first

    print("\n=== Accuracy when keeping lowest-uncertainty X% of samples ===")
    print("coverage\taccuracy\tn_kept")

    for cov in coverages:
        n_kept = int(round(num_samples * cov))
        if n_kept <= 0:
            continue
        idx_keep = sorted_idx[:n_kept]
        acc = float((y_pred[idx_keep] == y_true[idx_keep]).mean())
        print(f"{cov:.2f}\t\t{acc:.4f}\t{n_kept}")


if __name__ == "__main__":
    main()

