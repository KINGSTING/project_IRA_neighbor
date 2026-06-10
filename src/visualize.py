#!/usr/bin/env python3
"""
Visualization for POL-SPILL: Causal Graph VAE Results
Generates publication-ready maps, network graphs, and cluster plots.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
from adjustText import adjust_text

# ============================================================
# CONFIGURATION & STYLING
# ============================================================
DATA_DIR = "/home/jemarjohn/Documents/Research/project_IRA_neighbor/data"
OUTPUT_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Apply global publication styling
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
effects = np.load(os.path.join(DATA_DIR, "causal_effects.npz"), allow_pickle=True)
direct_effect = effects['direct_effect']                 
spill_export = effects['spillover_exported']             
spill_receive = effects['spillover_received']            
spill_matrix = effects['spillover_matrix']               
d_confounder = effects['d_confounder']                   
provinces = effects['provinces']                         

# Load and align geometry
gdf = gpd.read_file(os.path.join(DATA_DIR, "provinces_with_nodeid.geojson"))
gdf = gdf.sort_values('node_id')  
assert len(gdf) == len(provinces), "Mismatch between geometry and effects"

gdf['province'] = provinces
gdf['direct_effect'] = direct_effect
gdf['spillover_export'] = spill_export
gdf['spillover_receive'] = spill_receive
gdf['net_spillover'] = spill_export - spill_receive

# ============================================================
# 1. Choropleth Map: Direct Effect
# ============================================================
print("Generating Figure 1: Direct Effect Map...")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf.plot(column='direct_effect', cmap='coolwarm', legend=True,
         legend_kwds={'label': 'Direct Effect Magnitude', 'shrink': 0.7, 'orientation': 'vertical'},
         edgecolor='black', linewidth=0.3, ax=ax)

ax.set_title("Direct Fiscal-Electoral Effect by Province", fontsize=14, fontweight='bold', pad=15)
ax.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_direct_effect_map.png"))
plt.close()

# ============================================================
# 2. Top Exporters and Importers (Bar Charts)
# ============================================================
print("Generating Figure 2: Spillover Bars...")
export_df = pd.DataFrame({'province': provinces, 'export': spill_export}).nlargest(10, 'export')
import_df = pd.DataFrame({'province': provinces, 'import': spill_receive}).nlargest(10, 'import')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.despine()

sns.barplot(data=export_df, x='export', y='province', ax=axes[0], palette="Reds_r")
axes[0].set_title('Top 10 Volatility Exporters', fontweight='bold')
axes[0].set_xlabel('Spillover Effect (Exported)')
axes[0].set_ylabel('')

sns.barplot(data=import_df, x='import', y='province', ax=axes[1], palette="Blues_r")
axes[1].set_title('Top 10 Volatility Importers', fontweight='bold')
axes[1].set_xlabel('Spillover Effect (Received)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_spillover_bars.png"))
plt.close()

# ============================================================
# 3. Directed Spillover Network (Top 5% strongest links)
# ============================================================
print("Generating Figure 3: Spillover Network...")
G = nx.DiGraph()
threshold = np.percentile(spill_matrix, 95)
for i in range(len(provinces)):
    for j in range(len(provinces)):
        if spill_matrix[i, j] > threshold and i != j:
            G.add_edge(provinces[i], provinces[j], weight=spill_matrix[i, j])

gdf_centroids = gdf.copy()
gdf_centroids['centroid'] = gdf_centroids.geometry.centroid
pos = {row['province']: (row['centroid'].x, row['centroid'].y) for _, row in gdf_centroids.iterrows()}

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf.plot(ax=ax, facecolor='#eaeaea', edgecolor='white', linewidth=0.5)

# Use curved edges (arc3) for better visibility of bidirectional flows
nx.draw_networkx_nodes(G, pos, node_size=40, node_color='#2c3e50', edgecolors='white', ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='#e74c3c', alpha=0.7, width=1.2, 
                       arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.2", ax=ax)

ax.set_title("Directed Spillover Network (Top 5% Strongest Effects)", fontsize=14, fontweight='bold')
ax.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_spillover_network.png"))
plt.close()

# ============================================================
# 4. UMAP of Confounder Embeddings
# ============================================================
print("Generating Figure 4: Confounder UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
conf_2d = reducer.fit_transform(d_confounder)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(conf_2d[:, 0], conf_2d[:, 1], c=direct_effect, cmap='coolwarm', s=80, alpha=0.9, edgecolor='k', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax, label='Direct Effect Magnitude')

ax.set_title('Latent Confounder Space (UMAP)', fontweight='bold', pad=15)
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')

# Add province labels with smart adjustment to prevent overlap
texts = [ax.text(conf_2d[i, 0], conf_2d[i, 1], prov, fontsize=7) for i, prov in enumerate(provinces)]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), expand_points=(1.2, 1.2))

sns.despine()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_confounder_umap.png"))
plt.close()

# ============================================================
# 5. Spillover Matrix Heatmap
# ============================================================
print("Generating Figure 5: Spillover Heatmap...")
gdf['net'] = spill_export - spill_receive
top20 = gdf.nlargest(20, 'net')['province'].tolist()
idx_map = {prov: i for i, prov in enumerate(provinces)}
top_idx = [idx_map[p] for p in top20]
sub_matrix = spill_matrix[np.ix_(top_idx, top_idx)]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(sub_matrix, xticklabels=top20, yticklabels=top20, cmap='viridis', 
            square=True, cbar_kws={"shrink": .8, "label": "Spillover Intensity"}, ax=ax)

ax.set_title('Spillover Matrix (Top 20 Net Exporters)', fontweight='bold', pad=15)
ax.set_xlabel('Target Province (Importer)')
ax.set_ylabel('Source Province (Exporter)')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, "fig5_spillover_heatmap.png"))
plt.close()

print("All publication-ready figures generated successfully.")