#!/usr/bin/env python3
"""
Visualization for POL-SPILL: Causal Graph VAE Results
Generates publication-ready maps, network graphs, and cluster plots.
Figure 1 implements a premium geographical choropleth matching the style of Project Bahura.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
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

# Keep the original geographic CRS (EPSG:4326) for proper map display
if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)
elif gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

# Calculate centroids using a projected CRS, then project back to EPSG:4326 degree space
gdf['centroid'] = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)

# ============================================================
# 1. True Choropleth Map (Project Bahura Style Layout)
# ============================================================
print("Generating Figure 1: Direct Effect Choropleth Map (Project Bahura Style)...")
fig, ax = plt.subplots(1, 1, figsize=(9, 11))

# Set clean white background context
ax.set_facecolor('white')

# Plot the main choropleth map using the matching ColorBrewer RdYlBu diverging profile
gdf.plot(column='direct_effect', 
         cmap='RdYlBu_r', 
         legend=True,
         legend_kwds={
             'label': 'Direct Effect Magnitude (higher = more anomalous)', 
             'shrink': 0.75, 
             'orientation': 'vertical',
             'pad': 0.03
         },
         edgecolor='black', 
         linewidth=0.25, 
         ax=ax,
         zorder=1)

# Extract and map the exact geographic coordinate paths for centroid overlays
x_coords = gdf['centroid'].x
y_coords = gdf['centroid'].y
ax.scatter(x_coords, y_coords, color='black', s=8, zorder=3, alpha=0.6)

# Mirror explicit framing from Project Bahura
ax.set_title("Spatial Distribution of Direct Fiscal-Electoral Effects", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)

# Set map canvas boundaries to focus clean framing over the Philippine archipelago
ax.set_xlim(116, 127)
ax.set_ylim(4, 22)

# Re-enable the structural border frame lines around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.5)

plt.savefig(os.path.join(OUTPUT_DIR, "fig1_direct_effect_map.png"))
plt.close()
print("  Saved fig1_direct_effect_map.png")

# ============================================================
# 2. Top Exporters and Importers (Bar Charts)
# ============================================================
print("Generating Figure 2: Spillover Bars...")
export_df = pd.DataFrame({'province': provinces, 'export': spill_export}).nlargest(10, 'export')
import_df = pd.DataFrame({'province': provinces, 'import': spill_receive}).nlargest(10, 'import')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.despine()
sns.barplot(data=export_df, x='export', y='province', ax=axes[0], hue='province', palette="Reds_r", legend=False)
axes[0].set_title('Top 10 Volatility Exporters', fontweight='bold')
axes[0].set_xlabel('Spillover Effect (Exported)')
axes[0].set_ylabel('')

sns.barplot(data=import_df, x='import', y='province', ax=axes[1], hue='province', palette="Blues_r", legend=False)
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
            
pos = {row['province']: (row['centroid'].x, row['centroid'].y) for _, row in gdf.iterrows()}

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.3, alpha=0.5, zorder=1)
nx.draw_networkx_nodes(G, pos, node_size=40, node_color='black', ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='red', alpha=0.85, width=1.2,
                       arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.2", ax=ax)
ax.set_title("Directed Spillover Network (Top 5% Strongest Effects)", fontsize=14, fontweight='bold')
ax.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_spillover_network.png"))
plt.close()

# ============================================================
# 4. UMAP of Confounder Embeddings
# ============================================================
print("Generating Figure 4: Confounder UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, n_jobs=1)
conf_2d = reducer.fit_transform(d_confounder)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(conf_2d[:, 0], conf_2d[:, 1], c=direct_effect, cmap='coolwarm', s=80, alpha=0.9, edgecolor='k', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax, label='Direct Effect Magnitude')
ax.set_title('Latent Confounder Space (UMAP)', fontweight='bold', pad=15)
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
texts = [ax.text(conf_2d[i, 0], conf_2d[i, 1], prov, fontsize=7) for i, prov in enumerate(provinces)]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=10), expand_points=(1.2, 1.2))
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
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_spillover_heatmap.png"))
plt.close()

# ============================================================
# 6. Placebo Test Histogram (Figure 8)
# ============================================================
print("Generating Figure 8: Placebo Test Histograms...")

# Load placebo results
placebo_path = os.path.join(DATA_DIR, "placebo_results.npz")
if os.path.exists(placebo_path):
    placebo_data = np.load(placebo_path, allow_pickle=True)
    placebo_means = placebo_data['placebo_means']
    placebo_manila = placebo_data['placebo_manila']
    true_mean_abs = placebo_data['true_mean_abs']
    true_manila_bulacan = placebo_data['true_manila_bulacan']
    ci_mean = placebo_data['ci_mean']
    ci_manila = placebo_data['ci_manila']
    
    # Create two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Mean absolute spillover
    ax1 = axes[0]
    ax1.hist(placebo_means, bins=10, color='lightgray', edgecolor='black', alpha=0.7)
    ax1.axvline(true_mean_abs, color='red', linestyle='--', linewidth=2, label=f'True = {true_mean_abs:.3f}')
    ax1.axvline(ci_mean[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axvline(ci_mean[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Mean Absolute Spillover Magnitude')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Average Spillover', fontweight='bold')
    ax1.legend()
    
    # Panel 2: Manila → Bulacan spillover
    ax2 = axes[1]
    ax2.hist(placebo_manila, bins=10, color='lightgray', edgecolor='black', alpha=0.7)
    ax2.axvline(true_manila_bulacan, color='red', linestyle='--', linewidth=2, label=f'True = {true_manila_bulacan:.3f}')
    ax2.axvline(ci_manila[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axvline(ci_manila[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Spillover Magnitude (Manila → Bulacan)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Headline Asymmetric Link', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_placebo_histogram.png"))
    plt.close()
    print("  Saved fig8_placebo_histogram.png")
else:
    print("  Placebo results not found; skipping Figure 8.")

print("All publication-ready figures generated successfully.")