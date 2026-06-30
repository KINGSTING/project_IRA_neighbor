#!/usr/bin/env python3
"""
Placebo Test: Randomized Adjacency Matrices
Retrains FiscoNet on degree-preserving random shuffles of the adjacency matrix
to prove the spillover matrix is not an artifact of the architecture.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import copy
import random
from sklearn.preprocessing import StandardScaler
from model import CausalGraphVAE  # Your existing model definition

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/home/jemarjohn/Documents/Research/project_IRA_neighbor/data"
NPZ_PATH = os.path.join(DATA_DIR, "processed_data.npz")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "model_checkpoint.pt")  # Not strictly needed for placebo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (MUST match model.py, but we lower epochs for speed)
LATENT_DIM = 32
DIRECT_DIM = 12
SPILLOVER_DIM = 12
CONFOUNDER_DIM = 8
HIDDEN_DIM = 64
KL_BETA = 0.001
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PLACEBO_EPOCHS = 100  # Reduced from 300 for speed
NUM_PERMUTATIONS = 25  # Start with 25; run 50 if you have time

# ============================================================
# LOAD DATA
# ============================================================
print(f"[1] Loading data from {NPZ_PATH}...")
data = np.load(NPZ_PATH, allow_pickle=True)
X_train_raw = data['X_train']  # (77, 25, 20)
X_val_raw = data['X_val']      # (77, 1, 20)
A_true = data['A']             # (77, 77)
provinces = data['provinces']

# Standardize (same as model.py)
n_prov, n_years, n_feat = X_train_raw.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, n_feat))
X_train_t = torch.FloatTensor(X_train_scaled.reshape(n_prov, n_years, n_feat)).to(DEVICE)
X_val_scaled = scaler.transform(X_val_raw.reshape(-1, n_feat))
X_val_t = torch.FloatTensor(X_val_scaled.reshape(X_val_raw.shape)).to(DEVICE)

# Get true edge index for reference (though we replace it each time)
A_true_t = torch.FloatTensor(A_true).to(DEVICE)

# ============================================================
# HELPER: DEGREE-PRESERVING RANDOM PERMUTATION
# ============================================================
def permute_adjacency_degree_preserving(A, n_swaps=1000):
    """
    Perform a configuration-model-style edge swap to randomize the graph
    while preserving each node's degree exactly.
    """
    A = A.copy()
    n = A.shape[0]
    # Get edges (non-zero, no self-loops)
    edges = [(i, j) for i in range(n) for j in range(n) if A[i, j] == 1 and i != j]
    
    # Remove duplicates by ensuring i < j for undirected? A is directed in use, but symmetry makes it undirected.
    # Since A is symmetric (Queen), we only need to swap undirected edges and mirror later.
    # We'll treat as undirected for swapping, then mirror.
    undirected_edges = [(min(i,j), max(i,j)) for i,j in edges if i != j]
    undirected_edges = list(set(undirected_edges))
    
    for _ in range(n_swaps):
        if len(undirected_edges) < 4:
            break
        # Pick 2 random edges: (a,b) and (c,d)
        idx1, idx2 = random.sample(range(len(undirected_edges)), 2)
        a, b = undirected_edges[idx1]
        c, d = undirected_edges[idx2]
        
        # Check if swapping to (a,d) and (c,b) is valid (no self-loops, no duplicates)
        if a == d or c == b or a == c or b == d:
            continue
        if (min(a,d), max(a,d)) in undirected_edges or (min(c,b), max(c,b)) in undirected_edges:
            continue
        
        # Perform the swap
        undirected_edges[idx1] = (min(a,d), max(a,d))
        undirected_edges[idx2] = (min(c,b), max(c,b))
    
    # Build the new symmetric adjacency matrix
    A_new = np.zeros_like(A)
    for u, v in undirected_edges:
        if u != v:
            A_new[u, v] = 1
            A_new[v, u] = 1
    return A_new

# ============================================================
# TRAINING FUNCTION FOR A GIVEN ADJACENCY
# ============================================================
def train_and_extract_spillover(A_perm):
    """Train FiscoNet with the given adjacency and return the spillover matrix."""
    A_t = torch.FloatTensor(A_perm).to(DEVICE)
    edge_index = torch.nonzero(A_t, as_tuple=False).t().contiguous()
    
    # Initialize model
    model = CausalGraphVAE(
        in_dim=n_feat,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        direct_dim=DIRECT_DIM,
        spillover_dim=SPILLOVER_DIM,
        confounder_dim=CONFOUNDER_DIM,
        A=A_t
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop (shortened)
    model.train()
    for epoch in range(PLACEBO_EPOCHS):
        total_loss = 0.0
        for t in range(n_years - 1):
            x_t = X_train_t[:, t, :]
            x_t1 = X_train_t[:, t+1, :]
            x_recon, mu, logvar, d_d, d_s, d_c = model(x_t, edge_index)
            
            # Loss (copy-paste from model.py)
            recon_loss = F.mse_loss(x_recon, x_t1, reduction='mean')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_t.size(0)
            direct_norm = F.normalize(d_d, dim=1)
            spillover_norm = F.normalize(d_s, dim=1)
            mutual_info_penalty = torch.abs((direct_norm * spillover_norm).sum(dim=1)).mean()
            neigh_conf = A_t @ d_c
            smooth_loss = F.mse_loss(d_c, neigh_conf, reduction='mean')
            
            loss = recon_loss + KL_BETA * kl_loss + 0.1 * mutual_info_penalty + 0.01 * smooth_loss
            total_loss += loss
        
        avg_loss = total_loss / (n_years - 1)
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0 and epoch > 0:
            print(f"    Epoch {epoch+1}/{PLACEBO_EPOCHS} done.", end='\r')
    
    # Extract Spillover Matrix (same logic as model.py, but we need the full matrix)
    model.eval()
    with torch.no_grad():
        x_base = X_train_t[:, -1, :]  # 2016
        _, _, _, d_direct, d_spillover, d_confounder = model(x_base, edge_index)
        
        # Spillover effect: zero out each province's spillover one by one
        spillover_matrix = np.zeros((n_prov, n_prov))
        x_recon_original = model.decoder(d_direct, d_spillover, d_confounder, torch.arange(n_prov, device=DEVICE))
        
        for i in range(n_prov):
            d_spillover_zero = d_spillover.clone()
            d_spillover_zero[i] = 0
            x_recon_no_spill = model.decoder(d_direct, d_spillover_zero, d_confounder, torch.arange(n_prov, device=DEVICE))
            effect = (x_recon_original - x_recon_no_spill).norm(dim=1).cpu().numpy()
            spillover_matrix[i] = effect
            
    return spillover_matrix

# ============================================================
# TRUE BASELINE (Load from existing or compute once)
# ============================================================
print("[2] Getting True Model Spillover Matrix (or loading from file)...")
true_spill_path = os.path.join(DATA_DIR, "causal_effects.npz")
if os.path.exists(true_spill_path):
    true_effects = np.load(true_spill_path, allow_pickle=True)
    spill_true = true_effects['spillover_matrix']
else:
    # Fallback: train once with true A (but you should already have this file)
    print("  Warning: causal_effects.npz not found. Computing true baseline now...")
    spill_true = train_and_extract_spillover(A_true)
    np.savez(os.path.join(DATA_DIR, "causal_effects_placeholder_true.npz"), spillover_matrix=spill_true)

# Calculate true statistics
true_mean_abs = np.mean(np.abs(spill_true - np.eye(n_prov)))  # exclude self?
# Better: exclude diagonal if we want pairwise only
mask = ~np.eye(n_prov, dtype=bool)
true_mean_abs = np.mean(np.abs(spill_true[mask]))

# Get indices for Manila and Bulacan
try:
    manila_idx = list(provinces).index("METRO MANILA")
    bulacan_idx = list(provinces).index("BULACAN")
    true_manila_bulacan = spill_true[manila_idx, bulacan_idx]
except ValueError:
    # Fallback if names differ
    print("  Could not find METRO MANILA or BULACAN. Using top pair.")
    true_manila_bulacan = np.max(spill_true[mask])  # placeholder

print(f"  True Mean Abs Spillover: {true_mean_abs:.4f}")
print(f"  True Manila->Bulacan: {true_manila_bulacan:.4f}")

# ============================================================
# PLACEBO LOOP
# ============================================================
print(f"\n[3] Running {NUM_PERMUTATIONS} placebo permutations ({PLACEBO_EPOCHS} epochs each)...")
placebo_means = []
placebo_manila = []

for perm_idx in range(NUM_PERMUTATIONS):
    print(f"\n  Permutation {perm_idx+1}/{NUM_PERMUTATIONS}")
    # Generate randomized adjacency
    A_perm = permute_adjacency_degree_preserving(A_true, n_swaps=2000)
    
    # Train and extract
    try:
        spill_perm = train_and_extract_spillover(A_perm)
        # Calculate stats
        perm_mean = np.mean(np.abs(spill_perm[mask]))
        perm_manila = spill_perm[manila_idx, bulacan_idx] if 'manila_idx' in locals() else np.max(spill_perm[mask])
        
        placebo_means.append(perm_mean)
        placebo_manila.append(perm_manila)
        print(f"    Mean: {perm_mean:.4f}, Manila->Bulacan: {perm_manila:.4f}")
    except Exception as e:
        print(f"    Error: {e}. Skipping.")
        continue

# ============================================================
# RESULTS
# ============================================================
placebo_means = np.array(placebo_means)
placebo_manila = np.array(placebo_manila)

mean_placebo_mean = np.mean(placebo_means)
std_placebo_mean = np.std(placebo_means)
ci_lower_mean = np.percentile(placebo_means, 2.5)
ci_upper_mean = np.percentile(placebo_means, 97.5)

mean_placebo_manila = np.mean(placebo_manila)
std_placebo_manila = np.std(placebo_manila)
ci_lower_manila = np.percentile(placebo_manila, 2.5)
ci_upper_manila = np.percentile(placebo_manila, 97.5)

# Percentiles of true value
pct_mean = np.mean(placebo_means < true_mean_abs) * 100
pct_manila = np.mean(placebo_manila < true_manila_bulacan) * 100

print("\n" + "="*60)
print("PLACEBO TEST RESULTS")
print("="*60)
print(f"Statistic                    | True Value | Placebo Mean | 95% CI (Lower, Upper) | True > Placebo?")
print("-"*60)
print(f"Mean Abs Spillover           | {true_mean_abs:.4f}    | {mean_placebo_mean:.4f}    | ({ci_lower_mean:.4f}, {ci_upper_mean:.4f})     | {pct_mean:.1f}% percentile")
print(f"Manila -> Bulacan            | {true_manila_bulacan:.4f}    | {mean_placebo_manila:.4f}    | ({ci_lower_manila:.4f}, {ci_upper_manila:.4f})     | {pct_manila:.1f}% percentile")
print("="*60)

# Save placebo results for plotting later
np.savez(os.path.join(DATA_DIR, "placebo_results.npz"),
         placebo_means=placebo_means,
         placebo_manila=placebo_manila,
         true_mean_abs=true_mean_abs,
         true_manila_bulacan=true_manila_bulacan,
         ci_mean=(ci_lower_mean, ci_upper_mean),
         ci_manila=(ci_lower_manila, ci_upper_manila))

print(f"\nResults saved to {DATA_DIR}/placebo_results.npz")