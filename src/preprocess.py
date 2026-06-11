#!/usr/bin/env python3
"""
Phase 0: Data Preparation for POL-SPILL Project
Reads Excel file with full fiscal-electoral panel.
Guarantees Polygon geometry extraction for true map backgrounds.
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
# FIX: Updated file name to match your actual directory contents
PROV_SHAPEFILE_ZIP = os.path.join(ADMIN_DIR, "PH_Adm2_ProvDists.shp.zip")

OUTPUT_NPZ = os.path.join(DATA_DIR, "processed_data.npz")
OUTPUT_GEOJSON = os.path.join(DATA_DIR, "provinces_with_nodeid.geojson")

TRAIN_END_YEAR = 2016
VAL_YEARS = [2019]
TEST_YEARS = [2022]

FEATURE_COLS = [
    'lgusincome', 'totlocsrc', 'ira', 'tottax', 'totexsrc',
    'govexp', 'pubwelf', 'healthexp', 'econdevexp', 'totexp',
    'votes', 'pi', 'p1', 'ENC_lt', 'ENC_gol', 'loc_tot',
    'logpubwelf', 'loggovexp', 'logira', 'ira_tot'
]

# ============================================================
# HELPER: Read polygon shapefile from zip securely
# ============================================================
def read_shapefile_from_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
        if not shp_files:
            raise FileNotFoundError("No .shp file found in zip archive.")
        
        shp_path = None
        # Robust check: Open each file briefly and verify it contains Polygons
        for f in shp_files:
            path = os.path.join(temp_dir, f)
            test_gdf = gpd.read_file(path, rows=5)
            if test_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).any():
                shp_path = path
                print(f"    Successfully selected Polygon layer: {f}")
                break
                
        if shp_path is None:
            print("    WARNING: No explicit Polygon layer discovered. Falling back to default.")
            shp_path = os.path.join(temp_dir, shp_files[0])
            
        gdf = gpd.read_file(shp_path)
        return gdf
    finally:
        shutil.rmtree(temp_dir)

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

numeric_cols = FEATURE_COLS + ['votes', 'pi', 'p1', 'ENC_lt', 'ENC_gol', 'loc_tot']
numeric_cols = [c for c in numeric_cols if c in fiscal_df.columns]
fiscal_df = clean_numeric(fiscal_df, numeric_cols)

if 'position' in fiscal_df.columns:
    fiscal_df = fiscal_df[fiscal_df['position'] == 'Governor'].copy()
if 'incumbent' in fiscal_df.columns and 'candidate' in fiscal_df.columns:
    fiscal_df = fiscal_df[fiscal_df['candidate'] == fiscal_df['incumbent']].copy()

fiscal_df = fiscal_df.drop_duplicates(subset=['lgu', 'year'])
fiscal_df = fiscal_df[['lgu', 'year'] + FEATURE_COLS].copy()

# ============================================================
# 2. LOAD PROVINCE BOUNDARIES
# ============================================================
print(f"[2] Loading province boundaries from {os.path.basename(PROV_SHAPEFILE_ZIP)}...")
full_gdf = read_shapefile_from_zip(PROV_SHAPEFILE_ZIP)

# FIX: Resilient filtering for Adm2 datasets that might lack 'admin_leve' column
if 'admin_leve' in full_gdf.columns:
    provinces_gdf = full_gdf[full_gdf['admin_leve'] == 2].copy()
else:
    print("    Note: 'admin_leve' column missing. Using full dataset (Adm2 partition assumed).")
    provinces_gdf = full_gdf.copy()

# FIX: Resilient naming resolution to support both 'adm2_name' and OCHA standard 'ADM2_EN'
if 'adm2_name' in provinces_gdf.columns:
    provinces_gdf['province_clean'] = provinces_gdf['adm2_name'].str.upper().str.strip()
elif 'ADM2_EN' in provinces_gdf.columns:
    provinces_gdf['province_clean'] = provinces_gdf['ADM2_EN'].str.upper().str.strip()
elif 'adm2_en' in provinces_gdf.columns:
    provinces_gdf['province_clean'] = provinces_gdf['adm2_en'].str.upper().str.strip()
else:
    raise KeyError(f"Could not resolve province name column. Available: {list(provinces_gdf.columns)}")

# Load CSV filtering reference
prov_names = pd.read_csv(PROV_CSV)
csv_name_col = 'adm2_en' if 'adm2_en' in prov_names.columns else [c for c in prov_names.columns if 'name' in c.lower() or 'en' in c.lower()][0]
prov_names['province_clean'] = prov_names[csv_name_col].str.upper().str.strip()

valid_provinces = set(prov_names[prov_names['geo_level'] == 'Prov']['province_clean'])
provinces_gdf = provinces_gdf[provinces_gdf['province_clean'].isin(valid_provinces)]
print(f"    Valid spatial entities matched: {len(provinces_gdf)} provinces.")

# ============================================================
# 3. STANDARDIZE AND MERGE
# ============================================================
print("[3] Standardizing data parameters...")
fiscal_df['province_clean'] = fiscal_df['lgu'].str.upper().str.strip()
fiscal_df = fiscal_df[fiscal_df['province_clean'].isin(provinces_gdf['province_clean'])]
fiscal_gdf = provinces_gdf.merge(fiscal_df, on='province_clean', how='inner')

available_cols = set(fiscal_gdf.columns)
feature_cols = [col for col in FEATURE_COLS if col in available_cols]

# ============================================================
# 4. ADJACENCY MATRIX
# ============================================================
print("[4] Building Queen contiguity adjacency matrix...")
unique_provs = fiscal_gdf.drop_duplicates('province_clean')[['province_clean', 'geometry']]
unique_provs = unique_provs.set_index('province_clean')
w = Queen.from_dataframe(unique_provs, use_index=True)
adj_matrix = w.full()[0].astype(np.float32)
sp.save_npz(os.path.join(DATA_DIR, 'adj_matrix.npz'), sp.csr_matrix(adj_matrix))

# ============================================================
# 5. CREATE FEATURE TENSOR & IMPUTE
# ============================================================
print("[5] Creating unified tensor panel...")
panel = fiscal_gdf[['province_clean', 'year'] + feature_cols].copy()
panel = panel.drop_duplicates(subset=['province_clean', 'year']).sort_values(['province_clean', 'year'])

provinces_list = panel['province_clean'].unique()
years_list = sorted(panel['year'].unique())
n_prov, n_years, n_features = len(provinces_list), len(years_list), len(feature_cols)

X = np.full((n_prov, n_years, n_features), np.nan)
for i, prov in enumerate(provinces_list):
    prov_data = panel[panel['province_clean'] == prov].set_index('year')
    for j, year in enumerate(years_list):
        if year in prov_data.index:
            X[i, j, :] = prov_data.loc[year, feature_cols].values.astype(float)

for i in range(n_prov):
    for f in range(n_features):
        X[i, :, f] = pd.Series(X[i, :, f]).interpolate(method='linear', limit_direction='both').fillna(0).values

# ============================================================
# 6. SPLIT AND SAVE
# ============================================================
print("[6] Finalizing pipeline sets...")
year_to_idx = {year: idx for idx, year in enumerate(years_list)}
train_years = [y for y in years_list if y <= TRAIN_END_YEAR]
val_idx = [year_to_idx[y] for y in years_list if y in VAL_YEARS]
test_idx = [year_to_idx[y] for y in years_list if y in TEST_YEARS]
train_idx = [year_to_idx[y] for y in train_years]

np.savez(OUTPUT_NPZ,
         X_train=X[:, train_idx, :], X_val=X[:, val_idx, :], X_test=X[:, test_idx, :],
         A=adj_matrix, provinces=provinces_list,
         years_train=train_years, years_val=VAL_YEARS, years_test=TEST_YEARS,
         feature_names=feature_cols)

unique_provs_gdf = unique_provs.reset_index()
unique_provs_gdf['node_id'] = range(n_prov)
unique_provs_gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
print("Phase 0 complete. TRUE Polygon geometries extracted successfully.")