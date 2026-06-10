#!/usr/bin/env python3
"""
Latent Traversal for POL-SPILL
Visualizes how changing each latent component (direct, spillover, confounder)
affects predicted fiscal-electoral features.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/home/jemarjohn/Documents/Research/project_IRA_neighbor/data"
NPZ_PATH = os.path.join(DATA_DIR, "processed_data.npz")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "model_checkpoint.pt")  # full model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Example provinces to visualize (choose 2-3 with different characteristics)
EXAMPLE_PROVINCES = ["METRO MANILA", "CEBU", "MISAMIS ORIENTAL"]  # adjust to actual names

# Features to plot (select 4-6 most interpretable)
FEATURES_TO_PLOT = ['ira', 'totlocsrc', 'pubwelf', 'pi']  # IRA, local revenue, public welfare, vote share

# ============================================================
# LOAD DATA AND MODEL
# ============================================================
# Load processed data to get feature scaler, province names, adjacency
print("[1] Loading processed data...")
data = np.load(NPZ_PATH, allow_pickle=True)
X_train = data['X_train']          # (n_prov, n_years, n_feat)
A = data['A']                      # (n_prov, n_prov)
provinces = data['provinces']      # list of strings
feature_names = data['feature_names']

# Standardize features to match training
n_prov, n_years, n_feat = X_train.shape
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, n_feat))

# Get indices of example provinces
prov_indices = [i for i, p in enumerate(provinces) if p in EXAMPLE_PROVINCES]
if not prov_indices:
    print("Warning: Example provinces not found. Using first three provinces.")
    prov_indices = list(range(min(3, n_prov)))

# Load model
print("[2] Loading model checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
config = checkpoint['config']
from model import CausalGraphVAE  # import the model class (assumes model.py is in same directory)
model = CausalGraphVAE(
    in_dim=config['in_dim'],
    hidden_dim=config['hidden_dim'],
    latent_dim=config['latent_dim'],
    direct_dim=config['direct_dim'],
    spillover_dim=config['spillover_dim'],
    confounder_dim=config['confounder_dim'],
    A=torch.FloatTensor(A).to(DEVICE)
).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Compute the latent components for the base year (2016)
X_base = torch.FloatTensor(X_train[:, -1, :]).to(DEVICE)  # 2016 features
edge_index = torch.nonzero(torch.FloatTensor(A).to(DEVICE), as_tuple=False).t().contiguous()
with torch.no_grad():
    _, _, _, d_direct, d_spillover, d_confounder = model(X_base, edge_index)
d_direct = d_direct.cpu().numpy()
d_spillover = d_spillover.cpu().numpy()
d_confounder = d_confounder.cpu().numpy()

# ============================================================
# TRAVERSAL FUNCTION
# ============================================================
def traverse_component(component_name, comp_index, prov_idx, steps=11, step_size=2.0):
    """
    component_name: 'direct', 'spillover', or 'confounder'
    comp_index: which dimension to vary (0 to dim-1)
    prov_idx: index of province to perturb
    steps: number of traversal steps
    step_size: range = [-step_size, +step_size] in standard deviations
    """
    # Get base latent for this province
    if component_name == 'direct':
        base = d_direct[prov_idx, comp_index].copy()
        all_latents = d_direct.copy()
    elif component_name == 'spillover':
        base = d_spillover[prov_idx, comp_index].copy()
        all_latents = d_spillover.copy()
    else:
        base = d_confounder[prov_idx, comp_index].copy()
        all_latents = d_confounder.copy()
    
    # Create traversal values (linearly spaced)
    values = np.linspace(-step_size, step_size, steps)
    decoded_features = []
    for val in values:
        # Perturb the selected component for this province
        new_latents = all_latents.copy()
        new_latents[prov_idx, comp_index] = base + val
        # Convert back to tensors
        if component_name == 'direct':
            d_direct_t = torch.FloatTensor(new_latents).to(DEVICE)
            d_spillover_t = torch.FloatTensor(d_spillover).to(DEVICE)
            d_confounder_t = torch.FloatTensor(d_confounder).to(DEVICE)
        elif component_name == 'spillover':
            d_direct_t = torch.FloatTensor(d_direct).to(DEVICE)
            d_spillover_t = torch.FloatTensor(new_latents).to(DEVICE)
            d_confounder_t = torch.FloatTensor(d_confounder).to(DEVICE)
        else:
            d_direct_t = torch.FloatTensor(d_direct).to(DEVICE)
            d_spillover_t = torch.FloatTensor(d_spillover).to(DEVICE)
            d_confounder_t = torch.FloatTensor(new_latents).to(DEVICE)
        # Decode
        with torch.no_grad():
            pred_features = model.decoder(d_direct_t, d_spillover_t, d_confounder_t,
                                          torch.arange(n_prov, device=DEVICE))
        pred_features = pred_features.cpu().numpy()
        decoded_features.append(pred_features[prov_idx, :])  # only this province's predicted features
    
    return values, np.array(decoded_features)

# ============================================================
# GENERATE PLOTS
# ============================================================
print("[3] Generating latent traversal plots...")
# For each example province, for each component, vary the first dimension (or any)
# We'll vary component index 0 for simplicity (most significant)
comp_idx = 0
for prov_idx, prov_name in zip(prov_indices, EXAMPLE_PROVINCES):
    fig, axes = plt.subplots(3, len(FEATURES_TO_PLOT), figsize=(15, 12))
    fig.suptitle(f"Latent Traversal for {prov_name}", fontsize=16)
    
    for i, comp_name in enumerate(['direct', 'spillover', 'confounder']):
        values, decoded = traverse_component(comp_name, comp_idx, prov_idx)
        # Invert scaling to get original feature units
        decoded_scaled = decoded  # predicted features are already standardized
        # We'll plot standardized values (easier to compare)
        for j, feat_name in enumerate(FEATURES_TO_PLOT):
            if feat_name not in feature_names:
                axes[i, j].text(0.5, 0.5, f"{feat_name} not in features", ha='center', va='center')
                continue
            feat_idx = list(feature_names).index(feat_name)
            ax = axes[i, j]
            ax.plot(values, decoded_scaled[:, feat_idx], 'o-', color='C{}'.format(i))
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax.set_title(f"{comp_name.upper()} → {feat_name}")
            ax.set_xlabel("Perturbation (σ)")
            ax.set_ylabel("Predicted feature (std)")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"traversal_{prov_name.replace(' ', '_')}.png"), dpi=150)
    plt.close()
    print(f"Saved traversal for {prov_name}")

print("Done. Figures saved to data/")