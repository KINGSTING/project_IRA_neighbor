#!/usr/bin/env python3
"""
Phase 2: Causal Graph VAE for POL-SPILL
Disentangles direct, spillover, and confounder effects.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/home/jemarjohn/Documents/Research/project_IRA_neighbor/data"
NPZ_PATH = os.path.join(DATA_DIR, "processed_data.npz")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
LATENT_DIM = 32          # Dimension of total latent vector z
DIRECT_DIM = 12          # Dimension for d_direct (affects only focal province)
SPILLOVER_DIM = 12       # Dimension for d_spillover (affects neighbors)
CONFOUNDER_DIM = 8       # Dimension for d_confounder (regional shocks)
HIDDEN_DIM = 64
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
KL_BETA = 0.001          # KL divergence weight (beta-VAE)

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
print("[1] Loading processed data...")
data = np.load(NPZ_PATH, allow_pickle=True)
X_train = data['X_train']      # (77, 25, 20)
X_val = data['X_val']          # (77, 1, 20)
X_test = data['X_test']        # (77, 1, 20)
A = data['A']                  # (77, 77)
provinces = data['provinces']
years_train = data['years_train']
feature_names = data['feature_names']

# Standardize features (per feature, across provinces and time)
n_prov, n_years, n_feat = X_train.shape
X_train_reshaped = X_train.reshape(-1, n_feat)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(n_prov, n_years, n_feat)

# Also standardize val/test with same scaler
X_val_reshaped = X_val.reshape(-1, n_feat)
X_val_scaled = scaler.transform(X_val_reshaped)
X_val = X_val_scaled.reshape(X_val.shape)

X_test_reshaped = X_test.reshape(-1, n_feat)
X_test_scaled = scaler.transform(X_test_reshaped)
X_test = X_test_scaled.reshape(X_test.shape)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train).to(DEVICE)   # (77, 25, 20)
X_val_t = torch.FloatTensor(X_val).to(DEVICE)
X_test_t = torch.FloatTensor(X_test).to(DEVICE)
A_t = torch.FloatTensor(A).to(DEVICE)               # (77, 77)

# Build graph for adjacency (same for all time steps)
# For GCN, we need edge_index from adjacency matrix
edge_index = torch.nonzero(A_t, as_tuple=False).t().contiguous()

# ============================================================
# CAUSAL GRAPH VAE MODEL
# ============================================================
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class CausalDisentangler(nn.Module):
    """Disentangles z into direct, spillover, and confounder components."""
    def __init__(self, latent_dim, direct_dim, spillover_dim, confounder_dim):
        super().__init__()
        self.direct_dim = direct_dim
        self.spillover_dim = spillover_dim
        self.confounder_dim = confounder_dim
        # Learnable transformation z -> d
        self.W = nn.Parameter(torch.randn(latent_dim, direct_dim + spillover_dim + confounder_dim))
        # Mutual information penalty will be applied in loss
    
    def forward(self, z):
        # z: (num_nodes, latent_dim)
        d = z @ self.W   # linear transformation
        d_direct = d[:, :self.direct_dim]
        d_spillover = d[:, self.direct_dim:self.direct_dim+self.spillover_dim]
        d_confounder = d[:, self.direct_dim+self.spillover_dim:]
        return d_direct, d_spillover, d_confounder

class Decoder(nn.Module):
    """Predicts next year's features using disentangled components."""
    def __init__(self, direct_dim, spillover_dim, confounder_dim, hidden_dim, out_dim, A):
        super().__init__()
        self.A = A  # adjacency matrix (for spillover aggregation)
        self.fc1 = nn.Linear(direct_dim + spillover_dim + confounder_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, d_direct, d_spillover, d_confounder, node_idx):
        """
        node_idx: tensor of node indices for which we predict (can be all)
        Spillover effect: aggregate d_spillover from neighbors
        """
        # Aggregate spillover from neighbors: A * d_spillover
        spillover_aggregated = self.A[node_idx] @ d_spillover  # (num_nodes, spillover_dim)
        # Concatenate all components
        combined = torch.cat([d_direct[node_idx], spillover_aggregated, d_confounder[node_idx]], dim=1)
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        out = self.fc_out(h)
        return out

class CausalGraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, direct_dim, spillover_dim, confounder_dim, A):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim, latent_dim)
        self.disentangler = CausalDisentangler(latent_dim, direct_dim, spillover_dim, confounder_dim)
        self.decoder = Decoder(direct_dim, spillover_dim, confounder_dim, hidden_dim, in_dim, A)
        self.A = A
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, edge_index):
        # x: (num_nodes, feat_dim) for current year
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        d_direct, d_spillover, d_confounder = self.disentangler(z)
        # Predict next year's features
        x_recon = self.decoder(d_direct, d_spillover, d_confounder, torch.arange(x.size(0), device=x.device))
        return x_recon, mu, logvar, d_direct, d_spillover, d_confounder

# ============================================================
# LOSS FUNCTION
# ============================================================
def loss_function(x, x_recon, mu, logvar, d_direct, d_spillover, d_confounder, A, beta=KL_BETA):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence (for VAE)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Causal disentanglement penalty: minimize mutual info between d_direct and d_spillover
    # Approximate by negative cosine similarity
    direct_norm = F.normalize(d_direct, dim=1)
    spillover_norm = F.normalize(d_spillover, dim=1)
    mutual_info_penalty = torch.abs((direct_norm * spillover_norm).sum(dim=1)).mean()
    
    # Spatial smoothness: neighboring provinces should have similar confounders
    neigh_conf = A @ d_confounder
    smooth_loss = F.mse_loss(d_confounder, neigh_conf, reduction='mean')
    
    total_loss = recon_loss + beta * kl_loss + 0.1 * mutual_info_penalty + 0.01 * smooth_loss
    return total_loss, recon_loss, kl_loss, mutual_info_penalty, smooth_loss

# ============================================================
# TRAINING LOOP
# ============================================================
print("[2] Initializing model...")
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

print("[3] Training...")
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    # Train on each year sequentially (predict next year)
    for t in range(n_years - 1):
        x_t = X_train_t[:, t, :]          # current year features
        x_t1 = X_train_t[:, t+1, :]       # next year features
        x_recon, mu, logvar, d_d, d_s, d_c = model(x_t, edge_index)
        loss, rl, kl, mi, sm = loss_function(x_t1, x_recon, mu, logvar, d_d, d_s, d_c, A_t)
        total_loss += loss
    avg_train_loss = total_loss / (n_years - 1)
    train_losses.append(avg_train_loss.item())
    
    # Validation (only one step from 2016 to 2019)
    model.eval()
    with torch.no_grad():
        x_val_in = X_train_t[:, -1, :]    # 2016 features
        x_val_true = X_val_t[:, 0, :]     # 2019 features
        x_val_recon, _, _, _, _, _ = model(x_val_in, edge_index)
        val_loss = F.mse_loss(x_val_recon, x_val_true).item()
        val_losses.append(val_loss)
    
    optimizer.zero_grad()
    avg_train_loss.backward()
    optimizer.step()
    scheduler.step(val_loss)
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss.item():.4f}, Val Loss = {val_loss:.4f}")

# ============================================================
# EXTRACT EFFECTS FOR INTERPRETATION
# ============================================================
print("[4] Extracting causal effects...")
model.eval()
with torch.no_grad():
    # Use last training year (2016) as base
    x_base = X_train_t[:, -1, :]
    _, _, _, d_direct, d_spillover, d_confounder = model(x_base, edge_index)
    
    # Direct effect: change in d_direct on predicted outcome
    # We'll compute sensitivity by perturbing d_direct
    d_direct_perturbed = d_direct + 0.1 * torch.randn_like(d_direct)
    x_recon_original = model.decoder(d_direct, d_spillover, d_confounder, torch.arange(n_prov, device=DEVICE))
    x_recon_perturbed = model.decoder(d_direct_perturbed, d_spillover, d_confounder, torch.arange(n_prov, device=DEVICE))
    direct_effect = (x_recon_perturbed - x_recon_original).norm(dim=1).cpu().numpy()
    
    # Spillover effect: change in neighbor d_spillover
    # Create a mask where only province i's spillover is set to zero, compute effect on neighbors
    spillover_effect_matrix = np.zeros((n_prov, n_prov))
    for i in range(n_prov):
        d_spillover_zero = d_spillover.clone()
        d_spillover_zero[i] = 0
        x_recon_no_spill = model.decoder(d_direct, d_spillover_zero, d_confounder, torch.arange(n_prov, device=DEVICE))
        effect = (x_recon_original - x_recon_no_spill).norm(dim=1).cpu().numpy()
        spillover_effect_matrix[i] = effect
    
    # Spillover received by each province (sum over incoming)
    spillover_received = spillover_effect_matrix.sum(axis=0)
    spillover_exported = spillover_effect_matrix.sum(axis=1)

# ============================================================
# SAVE RESULTS
# ============================================================
print("[5] Saving results...")
np.savez(os.path.join(DATA_DIR, "causal_effects.npz"),
         direct_effect=direct_effect,
         spillover_exported=spillover_exported,
         spillover_received=spillover_received,
         spillover_matrix=spillover_effect_matrix,
         d_direct=d_direct.cpu().numpy(),
         d_spillover=d_spillover.cpu().numpy(),
         d_confounder=d_confounder.cpu().numpy(),
         provinces=provinces)

print("Model training and effect extraction complete.")