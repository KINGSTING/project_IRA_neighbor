#!/usr/bin/env python3
"""
Disentanglement Evaluation for POL-SPILL
Compares full model vs. ablation (no MI penalty).
Computes MIG, DCI, SAP, BetaVAE score.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score, accuracy_score
from scipy.stats import entropy, pearsonr
from sklearn.ensemble import GradientBoostingClassifier

# ============================================================
# PATHS
# ============================================================
DATA_DIR = "/home/jemarjohn/Documents/Research/project_IRA_neighbor/data"
NPZ_PATH = os.path.join(DATA_DIR, "processed_data.npz")
EFFECTS_FULL = os.path.join(DATA_DIR, "causal_effects.npz")
EFFECTS_ABLATION = os.path.join(DATA_DIR, "causal_effects_ablation.npz")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def compute_mig(d_latent, factors, num_bins=20):
    n_comp = d_latent.shape[1]
    n_factors = factors.shape[1]
    mig_scores = []
    for k in range(n_comp):
        mutual_info = []
        d_vals = d_latent[:, k]
        # avoid constant components
        if np.std(d_vals) < 1e-6:
            mig_scores.append(0)
            continue
        d_bins = np.percentile(d_vals, np.linspace(0, 100, num_bins+1)[1:-1])
        d_disc = np.digitize(d_vals, bins=d_bins)
        for f in range(n_factors):
            f_vals = factors[:, f]
            if np.std(f_vals) < 1e-6:
                continue
            f_bins = np.percentile(f_vals, np.linspace(0, 100, num_bins+1)[1:-1])
            f_disc = np.digitize(f_vals, bins=f_bins)
            mi = mutual_info_score(d_disc, f_disc)
            mutual_info.append(mi)
        if len(mutual_info) == 0:
            mig_scores.append(0)
            continue
        mi_sorted = np.sort(mutual_info)[::-1]
        if len(mi_sorted) > 1:
            mig = (mi_sorted[0] - mi_sorted[1]) / np.log(n_factors)
        else:
            mig = mi_sorted[0] / np.log(n_factors)
        mig_scores.append(mig)
    return np.mean(mig_scores)

def compute_dci(d_latent, factors):
    n_comp = d_latent.shape[1]
    n_factors = factors.shape[1]
    scaler = StandardScaler()
    d_scaled = scaler.fit_transform(d_latent)
    
    # For each factor, train a classifier to predict it from latents
    # We'll build a coefficient matrix (n_factors, n_comp)
    coef_matrix = np.zeros((n_factors, n_comp))
    accuracies = []
    for f in range(n_factors):
        factor = factors[:, f]
        # Skip constant factors
        if len(np.unique(factor)) < 2:
            continue
        # Discretize into 5 bins
        bins = np.percentile(factor, np.linspace(0, 100, 6)[1:-1])
        if len(np.unique(bins)) < 1:
            target = np.digitize(factor, bins=bins) if len(bins) > 0 else np.zeros_like(factor)
        else:
            target = np.digitize(factor, bins=bins)
        # Use logistic regression with L1 (sparse)
        clf = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, C=0.1, random_state=42)
        clf.fit(d_scaled, target)
        coef_abs = np.abs(clf.coef_).max(axis=0)  # max over classes
        coef_matrix[f, :] = coef_abs
        acc = clf.score(d_scaled, target)
        accuracies.append(acc)
    
    # Disentanglement: for each component, compute entropy of its weight distribution across factors
    disent_scores = []
    for c in range(n_comp):
        weights = coef_matrix[:, c]
        if weights.sum() == 0:
            disent_scores.append(0)
            continue
        weights = weights / weights.sum()
        entropy_w = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(n_factors)
        disent = 1 - entropy_w / max_entropy
        disent_scores.append(disent)
    disentanglement = np.mean(disent_scores)
    
    # Completeness: for each factor, entropy of its weight distribution across components
    comp_scores = []
    for f in range(n_factors):
        weights = coef_matrix[f, :]
        if weights.sum() == 0:
            comp_scores.append(0)
            continue
        weights = weights / weights.sum()
        entropy_w = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(n_comp)
        comp = 1 - entropy_w / max_entropy
        comp_scores.append(comp)
    completeness = np.mean(comp_scores)
    
    informativeness = np.mean(accuracies)
    return disentanglement, completeness, informativeness

def compute_sap(d_latent, factors):
    n_comp = d_latent.shape[1]
    n_factors = factors.shape[1]
    corr_matrix = np.zeros((n_comp, n_factors))
    for i in range(n_comp):
        for j in range(n_factors):
            corr, _ = pearsonr(d_latent[:, i], factors[:, j])
            corr_matrix[i, j] = np.abs(corr)
    sap_per_comp = []
    for i in range(n_comp):
        sorted_corr = np.sort(corr_matrix[i, :])[::-1]
        if len(sorted_corr) > 1:
            sap_per_comp.append(sorted_corr[0] - sorted_corr[1])
        else:
            sap_per_comp.append(0)
    return np.mean(sap_per_comp)

def compute_beta_vae_score(d_latent, factors, num_iters=1000):
    """
    BetaVAE score: train a linear classifier to predict a randomly chosen factor
    from a randomly chosen latent dimension, using the method from Higgins et al.
    """
    n_samples = d_latent.shape[0]
    n_comp = d_latent.shape[1]
    n_factors = factors.shape[1]
    correct = 0
    for _ in range(num_iters):
        # Randomly select a factor
        f_idx = np.random.randint(0, n_factors)
        factor = factors[:, f_idx]
        # Discretize factor into 2 bins (above/below median)
        median = np.median(factor)
        target = (factor > median).astype(int)
        # Randomly select a latent dimension
        c_idx = np.random.randint(0, n_comp)
        latent = d_latent[:, c_idx].reshape(-1, 1)
        # Train linear classifier (LogisticRegression) on random subset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(latent, target, test_size=0.3, random_state=None)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        correct += accuracy_score(y_test, pred)
    return correct / num_iters

# ============================================================
# MAIN
# ============================================================
def evaluate_model(effects_path, model_name):
    print(f"\n--- Evaluating {model_name} ---")
    effects = np.load(effects_path, allow_pickle=True)
    d_all = np.concatenate([effects['d_direct'], effects['d_spillover'], effects['d_confounder']], axis=1)
    # Load factors from processed data
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_train = data['X_train']
    factors = X_train[:, -1, :]   # 2016 features
    print(f"  d_all shape: {d_all.shape}, factors shape: {factors.shape}")
    
    mig = compute_mig(d_all, factors)
    dci_d, dci_c, dci_i = compute_dci(d_all, factors)
    sap = compute_sap(d_all, factors)
    beta = compute_beta_vae_score(d_all, factors)
    return {"MIG": mig, "DCI-D": dci_d, "DCI-C": dci_c, "DCI-I": dci_i, "SAP": sap, "BetaVAE": beta}

if __name__ == "__main__":
    print("Loading data...")
    # Check which files exist
    results = {}
    if os.path.exists(EFFECTS_FULL):
        results["Full Model (with MI)"] = evaluate_model(EFFECTS_FULL, "Full Model")
    if os.path.exists(EFFECTS_ABLATION):
        results["Ablation (no MI)"] = evaluate_model(EFFECTS_ABLATION, "Ablation Model")
    
    print("\n" + "="*60)
    print("DISENTANGLEMENT METRICS COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'MIG':<8} {'DCI-D':<8} {'DCI-C':<8} {'DCI-I':<8} {'SAP':<8} {'BetaVAE':<8}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['MIG']:.4f}   {metrics['DCI-D']:.4f}   {metrics['DCI-C']:.4f}   {metrics['DCI-I']:.4f}   {metrics['SAP']:.4f}   {metrics['BetaVAE']:.4f}")
    print("="*60)
    
    # Also save as CSV for LaTeX table
    import pandas as pd
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(DATA_DIR, "disentanglement_metrics.csv"))
    print(f"\nMetrics saved to {DATA_DIR}/disentanglement_metrics.csv")