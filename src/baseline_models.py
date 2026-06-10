#!/usr/bin/env python3
"""
Baseline Models for POL-SPILL Comparison
- Panel Fixed Effects (FE) with province and year dummies (using sklearn)
- Simple MLP (no spatial info)
- Plain GCN (same encoder as our model, no disentangler)
- Spatial Lag Model (SAR) using spreg if available
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
RESULT_PATH = os.path.join(DATA_DIR, "baseline_results.csv")
DEVICE = torch.device('cpu')

# ============================================================
# LOAD DATA
# ============================================================
print("Loading processed data...")
data = np.load(NPZ_PATH, allow_pickle=True)
X_train = data['X_train']   # (77, 25, 20)
X_val = data['X_val']       # (77, 1, 20)
X_test = data['X_test']     # (77, 1, 20)
A = data['A']               # (77, 77)
feature_names = data['feature_names']

n_prov, n_years_train, n_feat = X_train.shape

# Prepare panel data for FE: use all province-year pairs (t -> t+1)
# We'll create a list of (current, next) for each province and year t (1992-2015)
current_list = []
next_list = []
prov_ids = []
year_ids = []
for p in range(n_prov):
    for t in range(n_years_train - 1):
        current_list.append(X_train[p, t, :])
        next_list.append(X_train[p, t+1, :])
        prov_ids.append(p)
        year_ids.append(t)  # relative year index
current = np.array(current_list)  # (77*24, 20)
next_vals = np.array(next_list)   # (77*24, 20)

# Validation: use 2016 (last train year) to predict 2019
val_current = X_train[:, -1, :]   # (77, 20)
val_target = X_val[:, 0, :]       # (77, 20)

# ============================================================
# 1. PANEL FIXED EFFECTS (using OneHot + LinearRegression)
# ============================================================
print("\n[1] Training Panel Fixed Effects model...")
# Create dummy variables for province and year
prov_dummies = pd.get_dummies(pd.Series(prov_ids), prefix='prov')
year_dummies = pd.get_dummies(pd.Series(year_ids), prefix='year')
# Convert current features to DataFrame with string column names
current_df = pd.DataFrame(current, columns=[f'x{i}' for i in range(n_feat)])
X_fe = pd.concat([prov_dummies, year_dummies, current_df], axis=1)
# Ensure all column names are strings
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
# Align columns
X_val_fe = X_val_fe.reindex(columns=X_fe.columns, fill_value=0)
pred_fe = model_fe.predict(X_val_fe)
mse_val_fe = mean_squared_error(val_target, pred_fe)
print(f"  Validation MSE (FE): {mse_val_fe:.4f}")

# ============================================================
# 2. SIMPLE MLP (no spatial info)
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

# Convert to tensors
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
        print(f"  Epoch {epoch+1}: Train Loss = {loss.item():.4f}")
model_mlp.eval()
with torch.no_grad():
    pred_val = model_mlp(X_val_t)
    mse_val_mlp = loss_fn(pred_val, y_val_t).item()
print(f"  Validation MSE (MLP): {mse_val_mlp:.4f}")

# ============================================================
# 3. PLAIN GCN (no disentanglement, but with spatial info)
# ============================================================
print("\n[3] Training Plain GCN (no disentanglement)...")
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

# Prepare data for GCN: use all training years sequentially (like the VAE)
X_full_train = torch.FloatTensor(X_train).to(DEVICE)  # (77, 25, 20)
A_t = torch.FloatTensor(A).to(DEVICE)
edge_index = torch.nonzero(A_t, as_tuple=False).t().contiguous()

model_gcn = PlainGCN(n_feat, 64, n_feat).to(DEVICE)
optimizer = torch.optim.Adam(model_gcn.parameters(), lr=1e-4)  # lower LR
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
        print(f"  Epoch {epoch+1}: Train Loss = {avg_loss.item():.4f}")

model_gcn.eval()
with torch.no_grad():
    val_pred_gcn = model_gcn(X_full_train[:, -1, :], edge_index)  # 2016 -> 2019
    mse_val_gcn = loss_fn(val_pred_gcn, torch.FloatTensor(val_target).to(DEVICE)).item()
print(f"  Validation MSE (Plain GCN): {mse_val_gcn:.4f}")

# ============================================================
# 4. SPATIAL LAG MODEL (SAR) using spreg (if available)
# ============================================================
try:
    import libpysal as lp
    from spreg import ML_Lag
    print("\n[4] Training Spatial Lag Model (SAR)...")
    # Build weight matrix from A
    w = lp.weights.W(lp.weights.util.full2W(A))
    # Use 2016 features to predict 2019 features (average over features)
    # SAR is univariate, so we predict each target feature separately and average MSE.
    mse_sar_list = []
    for i in range(n_feat):
        y = val_target[:, i]
        X_sar = val_current
        model_sar = ML_Lag(y, X_sar, w=w, method='full', name_y=f'y{i}', name_x=[f'x{j}' for j in range(n_feat)])
        pred_sar = model_sar.predy
        mse_sar_list.append(mean_squared_error(y, pred_sar))
    mse_val_sar = np.mean(mse_sar_list)
    print(f"  Validation MSE (SAR, averaged): {mse_val_sar:.4f}")
except ImportError:
    print("\n[4] Spatial Lag Model skipped (spreg not installed)")
    mse_val_sar = np.nan
except Exception as e:
    print(f"\n[4] SAR failed: {e}")
    mse_val_sar = np.nan

# ============================================================
# 5. OUR FULL MODEL (load validation loss from training)
# ============================================================
print("\n[5] Loading our full model's validation MSE...")
try:
    effects_full = np.load(os.path.join(DATA_DIR, "causal_effects.npz"), allow_pickle=True)
    val_losses = effects_full['val_losses']
    best_val_mse = min(val_losses)  # best validation MSE during training
    print(f"  Our model best validation MSE: {best_val_mse:.4f}")
except Exception as e:
    print(f"  Failed to load: {e}")
    best_val_mse = np.nan

# ============================================================
# SAVE RESULTS TABLE
# ============================================================
results = pd.DataFrame({
    'Model': ['Panel FE', 'MLP (no spatial)', 'Plain GCN', 'Spatial Lag (SAR)', 'Our Graph VAE (full)'],
    'Validation MSE': [mse_val_fe, mse_val_mlp, mse_val_gcn, mse_val_sar, best_val_mse]
})
results.to_csv(RESULT_PATH, index=False)
print("\n" + results.to_string())
print(f"\nResults saved to {RESULT_PATH}")