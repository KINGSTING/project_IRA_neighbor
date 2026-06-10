#!/usr/bin/env python3
"""
Phase 0: Data Preparation for POL-SPILL Project
Reads Excel file with full fiscal-electoral panel.
"""

import os
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal as lp
from libpysal.weights import Queen
import scipy.sparse as sp

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/home/jemarjohn/Documents/Research/project_IRA_neighbor/data"
ADMIN_DIR = os.path.join(DATA_DIR, "admin_boundaries")
FISCAL_DIR = os.path.join(DATA_DIR, "comelec_fiscal")

FISCAL_FILE = os.path.join(FISCAL_DIR, "COMELEC_fiscal+electoral_data_July 2025.xlsx")
PROV_CSV = os.path.join(ADMIN_DIR, "PH_Adm2_ProvDists.csv")
PROV_SHAPEFILE_ZIP = os.path.join(ADMIN_DIR, "phl_admin_boundaries.shp.zip")

OUTPUT_NPZ = os.path.join(DATA_DIR, "processed_data.npz")
OUTPUT_GEOJSON = os.path.join(DATA_DIR, "provinces_with_nodeid.geojson")

TRAIN_END_YEAR = 2016
VAL_YEARS = [2019]
TEST_YEARS = [2022]

# Full feature set (as per data dictionary)
FEATURE_COLS = [
    'lgusincome', 'totlocsrc', 'ira', 'tottax', 'totexsrc',
    'govexp', 'pubwelf', 'healthexp', 'econdevexp', 'totexp',
    'votes', 'pi', 'p1', 'ENC_lt', 'ENC_gol', 'loc_tot',
    'logpubwelf', 'loggovexp', 'logira', 'ira_tot'
]

# ============================================================
# HELPER: Read shapefile from zip
# ============================================================
def read_shapefile_from_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
        if not shp_files:
            raise FileNotFoundError("No .shp file found in zip archive.")
        shp_path = os.path.join(temp_dir, shp_files[0])
        gdf = gpd.read_file(shp_path)
        return gdf
    finally:
        shutil.rmtree(temp_dir)

# ============================================================
# HELPER: Clean numeric columns (remove commas, convert to float)
# ============================================================
def clean_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ============================================================
# 1. LOAD FISCAL-ELECTORAL DATA (Excel)
# ============================================================
print("[1] Loading fiscal-electoral data from Excel...")
fiscal_df = pd.read_excel(FISCAL_FILE, sheet_name=0)
print(f"    Loaded {len(fiscal_df)} rows.")
print(f"    Columns in fiscal data: {fiscal_df.columns.tolist()}")

# Clean numeric columns
numeric_cols = FEATURE_COLS + ['votes', 'pi', 'p1', 'ENC_lt', 'ENC_gol', 'loc_tot']
numeric_cols = [c for c in numeric_cols if c in fiscal_df.columns]
fiscal_df = clean_numeric(fiscal_df, numeric_cols)

# Filter to Governor position (there are also Vice-Governor, etc.)
if 'position' in fiscal_df.columns:
    fiscal_df = fiscal_df[fiscal_df['position'] == 'Governor'].copy()
    print(f"    Filtered to Governor rows: {len(fiscal_df)} rows.")
else:
    print("    WARNING: 'position' column not found. Using all rows (may include other positions).")

# Keep only the incumbent governor (candidate matches incumbent column)
if 'incumbent' in fiscal_df.columns and 'candidate' in fiscal_df.columns:
    fiscal_df = fiscal_df[fiscal_df['candidate'] == fiscal_df['incumbent']].copy()
    print(f"    Filtered to incumbent governor rows: {len(fiscal_df)} rows.")
else:
    print("    WARNING: 'incumbent' or 'candidate' column missing. Using all rows (may have duplicates per province-year).")

# Ensure one row per province per year (drop duplicates if any)
fiscal_df = fiscal_df.drop_duplicates(subset=['lgu', 'year'])
print(f"    After deduplication: {len(fiscal_df)} rows.")

# Keep only necessary columns: province name (lgu), year, and features
fiscal_df = fiscal_df[['lgu', 'year'] + FEATURE_COLS].copy()

# ============================================================
# 2. LOAD PROVINCE BOUNDARIES
# ============================================================
print("[2] Loading province boundaries from phl_admin_boundaries.shp.zip...")
full_gdf = read_shapefile_from_zip(PROV_SHAPEFILE_ZIP)
print(f"    Loaded {len(full_gdf)} rows with all admin levels.")

# Filter to provinces (admin_leve == 2)
if 'admin_leve' in full_gdf.columns:
    provinces_gdf = full_gdf[full_gdf['admin_leve'] == 2].copy()
    print(f"    Filtered to {len(provinces_gdf)} rows with admin_leve=2 (provinces).")
else:
    raise KeyError("Column 'admin_leve' not found.")

# Use adm2_name for province name
if 'adm2_name' in provinces_gdf.columns:
    provinces_gdf['province_clean'] = provinces_gdf['adm2_name'].str.upper().str.strip()
else:
    raise KeyError("Column 'adm2_name' not found.")

# Load CSV province list to filter
prov_names = pd.read_csv(PROV_CSV)
prov_names['province_clean'] = prov_names['adm2_en'].str.upper().str.strip()
valid_provinces = set(prov_names[prov_names['geo_level'] == 'Prov']['province_clean'])
provinces_gdf = provinces_gdf[provinces_gdf['province_clean'].isin(valid_provinces)]
print(f"    After matching with CSV province list: {len(provinces_gdf)} provinces.")

# ============================================================
# 3. STANDARDIZE PROVINCE NAMES IN FISCAL DATA
# ============================================================
print("[3] Standardizing province names in fiscal data...")
fiscal_df['province_clean'] = fiscal_df['lgu'].str.upper().str.strip()

# Keep only provinces present in geometry
fiscal_df = fiscal_df[fiscal_df['province_clean'].isin(provinces_gdf['province_clean'])]
print(f"    Fiscal data now has {len(fiscal_df)} rows after matching geometry provinces.")

# ============================================================
# 4. MERGE FISCAL DATA WITH GEOMETRY
# ============================================================
print("[4] Merging fiscal data with geometry...")
fiscal_gdf = provinces_gdf.merge(
    fiscal_df,
    on='province_clean',
    how='inner'
)
print(f"    Merged GeoDataFrame has {len(fiscal_gdf)} rows.")

# ============================================================
# 5. HANDLE MISSING FEATURES
# ============================================================
available_cols = set(fiscal_gdf.columns)
desired_set = set(FEATURE_COLS)
feature_cols = [col for col in FEATURE_COLS if col in available_cols]
missing = desired_set - available_cols
if missing:
    print(f"    WARNING: Missing desired feature columns: {missing}")
    print(f"    Will proceed with available: {feature_cols}")
if not feature_cols:
    raise ValueError("No desired feature columns found. Check Excel file contents.")

# ============================================================
# 6. ADJACENCY MATRIX
# ============================================================
print("[5] Building Queen contiguity adjacency matrix...")
unique_provs = fiscal_gdf.drop_duplicates('province_clean')[['province_clean', 'geometry']]
unique_provs = unique_provs.set_index('province_clean')
w = Queen.from_dataframe(unique_provs, use_index=True)
adj_matrix = w.full()[0].astype(np.float32)
print(f"    Adjacency matrix shape: {adj_matrix.shape}")
sp.save_npz(os.path.join(DATA_DIR, 'adj_matrix.npz'), sp.csr_matrix(adj_matrix))

# ============================================================
# 7. CREATE FEATURE TENSOR
# ============================================================
print("[6] Creating feature tensor...")
panel = fiscal_gdf[['province_clean', 'year'] + feature_cols].copy()
panel = panel.drop_duplicates(subset=['province_clean', 'year'])
panel = panel.sort_values(['province_clean', 'year'])

provinces_list = panel['province_clean'].unique()
years_list = sorted(panel['year'].unique())
n_prov = len(provinces_list)
n_years = len(years_list)
n_features = len(feature_cols)
print(f"    Provinces: {n_prov}, Years: {n_years}, Features: {n_features}")

X = np.full((n_prov, n_years, n_features), np.nan)
for i, prov in enumerate(provinces_list):
    prov_data = panel[panel['province_clean'] == prov].set_index('year')
    for j, year in enumerate(years_list):
        if year in prov_data.index:
            vals = prov_data.loc[year, feature_cols].values.astype(float)
            X[i, j, :] = vals

# ============================================================
# 8. MISSING DATA IMPUTATION
# ============================================================
print("[7] Handling missing data with linear interpolation...")
for i in range(n_prov):
    for f in range(n_features):
        series = pd.Series(X[i, :, f])
        series = series.interpolate(method='linear', limit_direction='both').fillna(0)
        X[i, :, f] = series.values
print("    Missing data imputed.")

# ============================================================
# 9. TRAIN/VAL/TEST SPLIT
# ============================================================
print("[8] Splitting data by year...")
year_to_idx = {year: idx for idx, year in enumerate(years_list)}
train_years = [y for y in years_list if y <= TRAIN_END_YEAR]
val_years = [y for y in years_list if y in VAL_YEARS]
test_years = [y for y in years_list if y in TEST_YEARS]
train_idx = [year_to_idx[y] for y in train_years]
val_idx = [year_to_idx[y] for y in val_years]
test_idx = [year_to_idx[y] for y in test_years]

X_train = X[:, train_idx, :]
X_val = X[:, val_idx, :]
X_test = X[:, test_idx, :]
print(f"    Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# ============================================================
# 10. SAVE
# ============================================================
print("[9] Saving processed data...")
np.savez(OUTPUT_NPZ,
         X_train=X_train, X_val=X_val, X_test=X_test,
         A=adj_matrix,
         provinces=provinces_list,
         years_train=train_years, years_val=val_years, years_test=test_years,
         feature_names=feature_cols)
print(f"    Saved to {OUTPUT_NPZ}")

unique_provs_gdf = unique_provs.reset_index()
unique_provs_gdf['node_id'] = range(n_prov)
unique_provs_gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
print(f"    Saved province geometry to {OUTPUT_GEOJSON}")

print("Phase 0 completed successfully.")