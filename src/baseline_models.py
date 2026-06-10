#!/usr/bin/env python3
"""
Baseline Models for POL-SPILL Comparison
- Panel Fixed Effects (FE) with province and year dummies
- Simple MLP (no spatial)
- Plain GCN (spatial but no disentanglement)
- Our full Graph VAE
All models use standardized features for fair comparison.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/home/jemarjohn/Documents/Research/project_IRA_neighbor/data"
NPZ_PATH = os.path.join(DATA_DIR, "processed_data.npz")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "model_checkpoint.pt")
RESULT_PATH = os.path.join(DATA_DIR, "baseline_results.csv")
DEVICE = torch.device('cpu')

# ============================================================
# LOAD AND STANDARDIZE DATA
# ============================================================
print("Loading processed data...")
data = np.load(NPZ_PATH, allow_pickle=True)
X_train = data['X_train']   # (77, 25, 20)
X_val = data['X_val']       # (77, 1, 20)
X_test = data['X_test']     # (77, 1, 20)
A = data['A']               # (77, 77)
provinces = data['provinces']
feature_names = data['feature_names']

n_prov, n_years_train, n_feat = X_train.shape

# Standardize features (same as model.py)
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, n_feat)
scaler.fit(X_train_flat)
X_train_norm = scaler.transform(X_train_flat).reshape(n_prov, n_years_train, n_feat)
X_val_norm = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
X_test_norm = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

# Prepare panel data for FE: current -> next for each province and year t (1992-2015)
current_list = []
next_list = []
prov_ids = []
year_ids = []
for p in range(n_prov):
    for t in range(n_years_train - 1):
        current_list.append(X_train_norm[p, t, :])
        next_list.append(X_train_norm[p, t+1, :])
        prov_ids.append(p)
        year_ids.append(t)
current = np.array(current_list)      # (77*24, 20)
next_vals = np.array(next_list)       # (77*24, 20)

# Validation: use 2016 (last train year) to predict 2019
val_current = X_train_norm[:, -1, :]  # (77, 20)
val_target = X_val_norm[:, 0, :]      # (77, 20)

# ============================================================
# 1. PANEL FIXED EFFECTS
# ============================================================
print("\n[1] Training Panel Fixed Effects model...")
prov_dummies = pd.get_dummies(pd.Series(prov_ids), prefix='prov')
year_dummies = pd.get_dummies(pd.Series(year_ids), prefix='year')
current_df = pd.DataFrame(current, columns=[f'x{i}' for i in range(n_feat)])
X_fe = pd.concat([prov_dummies, year_dummies, current_df], axis=1)
X_fe.columns = X_fe.columns.astype(str)
y_fe = pd.DataFrame(next_vals, columns=[f'y{i}' for i in range(n_feat)])

model_fe = LinearRegression()
model_fe.fit(X_fe, y_fe)

# Predict on validation
val_prov_dummies = pd.get_dummies(pd.Series(range(n_prov)), prefix='prov')
val_year_dummies = pd.get_dummies(pd.Series([n_years_train-1]*n_prov), prefix='year')
val_current_df = pd.DataFrame(val_current, columns=[f'x{i}' for i in range(n_feat)])
X_val_fe = pd.concat([val_prov_dummies, val_year_dummies, val_current_df], axis=1)
X_val_fe.columns = X_val_fe.columns.astype(str)
X_val_fe = X_val_fe.reindex(columns=X_fe.columns, fill_value=0)
pred_fe = model_fe.predict(X_val_fe)
mse_val_fe = mean_squared_error(val_target, pred_fe)
print(f"  Validation MSE (FE): {mse_val_fe:.6f}")

# ============================================================
# 2. MLP BASELINE
# ============================================================
print("\n[2] Training MLP baseline...")
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

X_train_t = torch.FloatTensor(current).to(DEVICE)
y_train_t = torch.FloatTensor(next_vals).to(DEVICE)
X_val_t = torch.FloatTensor(val_current).to(DEVICE)
y_val_t = torch.FloatTensor(val_target).to(DEVICE)

model_mlp = MLP(n_feat, 128, n_feat).to(DEVICE)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(200):
    model_mlp.train()
    optimizer.zero_grad()
    pred = model_mlp(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_mlp.parameters(), 1.0)
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"  Epoch {epoch+1}: Train Loss = {loss.item():.6f}")
model_mlp.eval()
with torch.no_grad():
    pred_val = model_mlp(X_val_t)
    mse_val_mlp = loss_fn(pred_val, y_val_t).item()
print(f"  Validation MSE (MLP): {mse_val_mlp:.6f}")

# ============================================================
# 3. PLAIN GCN (spatial, no disentanglement)
# ============================================================
print("\n[3] Training Plain GCN...")
class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.conv2(x, edge_index)
        return x

X_full_train = torch.FloatTensor(X_train_norm).to(DEVICE)  # (77, 25, 20)
A_t = torch.FloatTensor(A).to(DEVICE)
edge_index = torch.nonzero(A_t, as_tuple=False).t().contiguous()

model_gcn = PlainGCN(n_feat, 64, n_feat).to(DEVICE)
optimizer = torch.optim.Adam(model_gcn.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(300):
    model_gcn.train()
    total_loss = 0.0
    for t in range(n_years_train - 1):
        x_t = X_full_train[:, t, :]
        x_t1 = X_full_train[:, t+1, :]
        pred = model_gcn(x_t, edge_index)
        loss = loss_fn(pred, x_t1)
        total_loss += loss
    avg_loss = total_loss / (n_years_train - 1)
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_gcn.parameters(), 1.0)
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"  Epoch {epoch+1}: Train Loss = {avg_loss.item():.6f}")

model_gcn.eval()
with torch.no_grad():
    val_pred_gcn = model_gcn(X_full_train[:, -1, :], edge_index)
    mse_val_gcn = loss_fn(val_pred_gcn, torch.FloatTensor(val_target).to(DEVICE)).item()
print(f"  Validation MSE (Plain GCN): {mse_val_gcn:.6f}")

# ============================================================
# 4. OUR FULL MODEL (load from checkpoint)
# ============================================================
print("\n[4] Loading our full model from checkpoint...")
# Import the model class from model.py
import importlib.util
spec = importlib.util.spec_from_file_location("model", "model.py")
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
CausalGraphVAE = model_module.CausalGraphVAE

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
config = checkpoint['config']
model_ours = CausalGraphVAE(
    in_dim=config['in_dim'],
    hidden_dim=config['hidden_dim'],
    latent_dim=config['latent_dim'],
    direct_dim=config['direct_dim'],
    spillover_dim=config['spillover_dim'],
    confounder_dim=config['confounder_dim'],
    A=A_t
).to(DEVICE)
model_ours.load_state_dict(checkpoint['model_state_dict'])
model_ours.eval()

# Evaluate on validation set
with torch.no_grad():
    x_val_in = torch.FloatTensor(X_train_norm[:, -1, :]).to(DEVICE)  # 2016
    y_val_true = torch.FloatTensor(val_target).to(DEVICE)
    x_val_recon, _, _, _, _, _ = model_ours(x_val_in, edge_index)
    mse_val_ours = loss_fn(x_val_recon, y_val_true).item()
print(f"  Validation MSE (Our Graph VAE): {mse_val_ours:.6f}")

# ============================================================
# SAVE RESULTS
# ============================================================
results = pd.DataFrame({
    'Model': ['Panel FE', 'MLP (no spatial)', 'Plain GCN', 'Our Graph VAE'],
    'Validation MSE': [mse_val_fe, mse_val_mlp, mse_val_gcn, mse_val_ours]
})
results.to_csv(RESULT_PATH, index=False)
print("\n" + results.to_string())
print(f"\nResults saved to {RESULT_PATH}")