# -*- coding: utf-8 -*-
"""
Central configuration for the decomposition pipeline.

Every script imports its paths from here, so the repository can be moved or
cloned without editing individual files. Raw inputs live outside the repository
and are located through environment variables; derived tables shipped under
``data/`` and all results written under ``outputs/`` are resolved automatically.

Environment variables (all optional):

    NDVI_MONTHLY   monthly NDVI GeoTIFFs of the whole region (YYYY-MM.tif)
    NDVI_CLASS     monthly NDVI GeoTIFFs per land-cover class
    LC_DIR         annual CLCD land-cover GeoTIFFs (30 m)
    TEMP_DIR       monthly mean temperature GeoTIFFs
    PRECIP_DIR     monthly total precipitation GeoTIFFs

Example::

    export NDVI_MONTHLY="/data/modis/monthly"
    python code/02_regional_decomposition.py
"""

import os

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs')

# --------------------------------------------------------------------------
# Raw inputs (not shipped with the repository)
# --------------------------------------------------------------------------
NDVI_MONTHLY = os.environ.get(
    'NDVI_MONTHLY', r'F:/JJJ/JJJ/JJJ数据/NDVI/月均NDVI')
NDVI_CLASS = os.environ.get(
    'NDVI_CLASS', r'G:/JJJ/NDVI/WH/月均')
LC_DIR = os.environ.get(
    'LC_DIR', r'G:/China/北方/土地利用WH')
TEMP_DIR = os.environ.get(
    'TEMP_DIR', r'G:/数据/月均温度/京津冀')
PRECIP_DIR = os.environ.get(
    'PRECIP_DIR', r'G:/数据/月均降水/京津冀')

# Sub-folders of NDVI_CLASS, one per land-cover class
CLASS_FOLDERS = {
    'forest': '森林',
    'grass': '草地',
    'crop': '农田',
    'impervious': '不透水地面',
}

# A reference raster that defines the common 1 km grid
NDVI_REFERENCE = os.path.join(NDVI_MONTHLY, '2001-01.tif')

# --------------------------------------------------------------------------
# Derived tables shipped with the repository
# --------------------------------------------------------------------------
CLASS_SERIES = os.path.join(DATA, 'class_ndvi_series_2001_2025.csv')
ANCHOR_SHARES = os.path.join(DATA, 'anchor_area_shares_percent.csv')

# --------------------------------------------------------------------------
# Intermediates and results
# --------------------------------------------------------------------------
GRID_CUBE_CACHE = os.path.join(OUT, 'grid_cube_cache.npz')
GRID_ATTRIBUTION = os.path.join(OUT, 'grid_attribution.npz')
GRID_CONTRIBUTION = os.path.join(OUT, 'grid_contribution.npz')
CLIMATE_ATTRIBUTION = os.path.join(OUT, 'climate_attribution.npz')
CLIMATE_CACHE = os.path.join(OUT, 'climate_cache.npz')

DECOMPOSITION_XLSX = os.path.join(OUT, 'decomposition_results.xlsx')
PERIOD_SUMMARY = os.path.join(OUT, 'period_summary.csv')
PERIOD_BY_CLASS = os.path.join(OUT, 'period_by_class_structural.csv')
CLIMATE_SENSITIVITY = os.path.join(OUT, 'climate_sensitivity.csv')
PERIOD_CLIMATE_SPLIT = os.path.join(OUT, 'period_climate_split.csv')
STABLE_PIXEL_CHECK = os.path.join(OUT, 'stable_pixel_check.csv')

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------
CLASSES = ['forest', 'grass', 'crop', 'impervious']
YEARS = list(range(2001, 2026))
ANCHOR_YEARS = [2001, 2005, 2010, 2015, 2020, 2025]


def ensure_output_dir():
    """Create ``outputs/`` if it does not exist and return its path."""
    os.makedirs(OUT, exist_ok=True)
    return OUT


def output(name):
    """Return an absolute path inside ``outputs/``, creating the folder."""
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)
