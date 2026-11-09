# Within-type and structural decomposition of regional NDVI trends

Code and derived data accompanying the manuscript:

> **Within-type and structural decomposition of regional NDVI trends: a case study of the
> northwestern Beijing–Tianjin–Hebei region**
> *Submitted to the International Journal of Applied Earth Observation and Geoinformation*

---

## What this code does

Regional mean NDVI is a composite signal: it rises both when vegetation within a land-cover
type becomes denser, and when area is reallocated toward greener types. Because the regional
mean is exactly the area-weighted sum of type-level greenness,

```
NDVI_reg(y) = Σ_k s_k(y) · NDVI_k(y)
```

its trend decomposes *without approximation* into

```
β_reg    = β_within + β_struct + β_int
β_within = Σ_k s̄_k · β_k                              (within-type greening)
β_struct = Σ_k NDVĪ_k · γ_k                           (land-use structural change)
β_int    = Σ_k Σ_y (y−ȳ)(s_k−s̄_k)(NDVI_k−NDVĪ_k) / Σ_y (y−ȳ)²
```

The within-type term is then split into a climatically explicable part and a non-climatic
residual by a pixel-wise NDVI ~ temperature + precipitation regression.

For the study region (2001–2025) the reported values are:

| Quantity | Value |
|---|---|
| Regional trend β_reg | 0.00316 a⁻¹ (R² = 0.80) |
| Within-type β_within | 0.00280 a⁻¹ (88%) |
| Structural β_struct | 0.00037 a⁻¹ (12%) |
| Climatic part of β_within | 0.00097 a⁻¹ (31% of β_reg) |
| Non-climatic residual | 0.00182 a⁻¹ (58% of β_reg) |
| Stable-pixel check | 0.00320 a⁻¹ (+14% over 0.00280) |

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── code/
│   ├── config.py                          shared path and constant definitions
│   ├── 01_build_grid_attribution_cube.py
│   ├── 02_regional_decomposition.py
│   ├── 03_grid_decomposition.py
│   ├── 04_climate_attribution.py
│   ├── 05_climate_sensitivity.py
│   ├── 06_stable_pixel_check.py
│   └── 07_counterfactual_scenarios.py
├── data/
│   ├── class_ndvi_series_2001_2025.csv
│   └── anchor_area_shares_percent.csv
└── outputs/                               written by the scripts; empty in the repository
```

All paths are defined once in `code/config.py`; no script contains an absolute path.

---

## Data

All three raw inputs are publicly available and are not redistributed here.

| Input | Source | Resolution / period |
|---|---|---|
| NDVI | MOD13Q1.061, retrieved through Google Earth Engine | 250 m, 16-day, 2001–2025 |
| Land cover | CLCD — Yang & Huang (2021), <https://doi.org/10.5194/essd-13-3907-2021> | 30 m, annual, 2000–2025 |
| Climate | Monthly mean temperature and precipitation for China | 0.008333°, 1901–2024 |

Two derived tables are shipped in `data/`, so the decomposition can be rerun without
reprocessing the rasters:

- **`class_ndvi_series_2001_2025.csv`** — monthly mean NDVI for each land-cover class and for
  the whole region. Columns: `date` (YYYY-MM), `type`
  (`forest` | `grass` | `crop` | `impervious` | `overall`), `mean`, `nvalid`.
- **`anchor_area_shares_percent.csv`** — area shares (%) of the four classes at the five-year
  anchor years 2001, 2005, 2010, 2015, 2020, 2025.

---

## Configuration

Raw inputs are located through environment variables; unset variables fall back to the
author's original paths.

| Variable | Meaning |
|---|---|
| `NDVI_MONTHLY` | monthly NDVI GeoTIFFs of the whole region (`YYYY-MM.tif`) |
| `NDVI_CLASS` | monthly NDVI GeoTIFFs, one sub-folder per class |
| `LC_DIR` | annual CLCD land-cover GeoTIFFs |
| `TEMP_DIR` | monthly mean temperature GeoTIFFs |
| `PRECIP_DIR` | monthly total precipitation GeoTIFFs |

```bash
export NDVI_MONTHLY=/data/modis/ndvi_monthly
export LC_DIR=/data/clcd
python code/02_regional_decomposition.py
```

---

## Workflow

```
02  regional decomposition ──────────────── Tables 1 and 2
07  counterfactual compositions ─────────── Table 3
        (both read only data/, no rasters needed)

01  build the 1 km per-pixel cube (NDVI + fractional cover, 2001–2025)
        │  → grid_attribution.npz
        ├── 03 grid decomposition ───────── Figure 4
        ├── 04 climate attribution ──────── Table 4 (main result)
        ├── 05 climate sensitivity ──────── Table 4 (sensitivity), Figure 5
        └── 06 stable-pixel check ───────── conservative bound (+14%)
```

`01` is the common upstream step for `03`–`06`; `02` and `07` are self-contained.

---

## Scripts

| # | Script | Purpose | Main output | Manuscript item |
|---|---|---|---|---|
| 01 | `build_grid_attribution_cube` | Resample 30 m CLCD to a 1 km fractional-cover grid, extract class-wise NDVI, write the per-pixel cube | `grid_attribution.npz` | Sections 2.2, 4.4 |
| 02 | `regional_decomposition` | Full-period, stage-wise and stage × class decomposition; OLS rate form and Bennett additive form | `decomposition_results.xlsx`, `period_summary.csv`, `period_by_class_structural.csv` | **Tables 1 and 2** |
| 03 | `grid_decomposition` | Aggregate-conserving per-pixel decomposition; the regional average reproduces Table 1 | `grid_contribution.npz` | Section 4.4 |
| 04 | `climate_attribution` | Pixel-wise NDVI ~ T + P regression; splits β_within into climatic and non-climatic parts | `climate_attribution.npz` | **Table 4** (main row) |
| 05 | `climate_sensitivity` | Three alternative climate specifications and a stage-wise climate split | `climate_sensitivity.csv`, `period_climate_split.csv`, `Fig_climate_attribution.*` | **Table 4**, **Figure 5** |
| 06 | `stable_pixel_check` | Recomputes β_within on pixels that stayed in the same class in 2001 and 2025 | `stable_pixel_check.csv` | Section 4.2.2 (+14%) |
| 07 | `counterfactual_scenarios` | Static effect of five alternative 2025 compositions | `counterfactual_scenarios.csv` | **Table 3** |

---

## Reproducing the headline numbers

Tables 1 and 2 need no raster processing; both input tables are in `data/`:

```bash
python code/02_regional_decomposition.py
```

Expected output:

```
β_reg    = 0.00317 a⁻¹
β_within = 0.00280 a⁻¹  (88.4%)
β_struct = 0.00037 a⁻¹  (11.6%)
β_int    = -2.4e-06 a⁻¹
identity check = -1.3e-18      (within + struct + int − reg)
```

Table 3 runs from the same two tables:

```bash
python code/07_counterfactual_scenarios.py
```

```
S1  3% grassland -> forest           +0.0039
S2  5% grassland -> forest           +0.0065
S3  5% forest -> grassland           -0.0065
S4  5% cropland -> forest            +0.0077
S5  2% cropland -> impervious        -0.0003
```

The pixel-level analyses (03–06) additionally require `grid_attribution.npz`, produced by
`01` from the raw rasters. Small differences in the fourth decimal arise from rounding in the
shipped area shares.

---

## Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

`01` processes 25 years of monthly rasters and is the memory-intensive step; it was run on a
machine with 16 GB of RAM. All other scripts operate on cached arrays and are fast.

---

## Notes

- **Anchoring.** Area shares are anchored at five-year intervals (2001, 2005, 2010, 2015,
  2020, 2025). Class means are computed with membership held fixed within each interval, so
  the class-mean series reflects greenness change rather than membership turnover.
- **Estimator.** The decomposition is exact only for a linear trend operator. OLS and the
  endpoint (Bennett) form both satisfy this; the median-based Theil–Sen slope does not
  preserve additivity and is therefore not used.
- **Climate split.** The split runs to 2024 because the climate data end there. The 2001–2024
  within-type term (0.00279 a⁻¹) is nearly identical to the 2001–2025 value (0.00280 a⁻¹), so
  the shares are carried over to the full period.
- **Scope.** Figure rendering scripts are not included; the repository covers the numerical
  results reported in the manuscript.
- **Language.** Comments in the scripts are in Chinese, as written during the analysis; all
  variable names, outputs and console tables are in English.

---

## Citation

If you use this code, please cite the manuscript above. A DOI will be added upon acceptance.

## License

Code released under the [MIT License](LICENSE).

The two tables in `data/` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The raw satellite and climate
products remain under their original providers' terms.
