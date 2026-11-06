# Feature Description
## Road Surface Condition Classification — FDIP Project

---

## Overview

Each road image is converted to a **33-dimensional feature vector** composed of four
interpretable, handcrafted descriptor groups. No learned or deep features are used.
The pipeline is fully deterministic.

| Group | Feature type | # Features |
|---|---|---:|
| A | Grayscale histogram | 16 |
| B | GLCM texture properties | 6 |
| C | LBP histogram (uniform) | 10 |
| D | Canny edge density | 1 |
| **Total** | | **33** |

---

## Pre-processing Pipeline

Applied identically to every image before any feature is computed.

| Step | Operation | Parameter | Rationale |
|---|---|---|---|
| 1 | Load RGB | — | Preserve colour for visualisation |
| 2 | Crop top 50% | `CROP_TOP_FRACTION = 0.50` | Remove sky, horizon, vehicles, buildings. Visual inspection of crop_comparison.png confirmed top-30% still retained cars and buildings in mid-frame; top-50% isolates the near-road surface reliably |
| 3 | Resize | `128 × 128 px` (Lanczos) | Consistent pixel count for histogram and GLCM |
| 4 | Convert to grayscale | `L` mode | Texture/edge features are intensity-based |
| 5 | Mild Gaussian blur | `radius = 1.0` (PIL) | Suppress JPEG noise before GLCM/LBP; not applied for Canny |

> **Two versions of the grayscale image are maintained internally:**
> - `gray_raw` (no blur) → used for histogram and Canny edge detection
> - `gray_blur` (blur applied) → used for GLCM and LBP

---

## Feature Group A — Grayscale Histogram (16 features)

**Column names:** `hist_00` … `hist_15`

### Method
- Compute a 16-bin histogram of pixel intensities over [0, 255]
- Normalise so all bins sum to 1.0

### Interpretation
- Smooth asphalt → concentrated in mid-to-high intensity bins (grey/bright)
- Unpaved/rough roads → wider spread, often darker tones
- Paving stones → may show bimodal distributions (joint gaps vs tile faces)

### Parameters
| Parameter | Value |
|---|---|
| Number of bins | 16 |
| Intensity range | [0, 255] |
| Normalisation | Sum = 1.0 |

---

## Feature Group B — GLCM Texture (6 features)

**Column names:** `glcm_contrast`, `glcm_dissimilarity`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`, `glcm_ASM`

### Method
1. Compute the Grey-Level Co-occurrence Matrix (GLCM) using `skimage.feature.graycomatrix`
2. Use **4 angles** (0°, 45°, 90°, 135°) and **distance = 1 pixel** (appropriate for 128×128 images)
3. Extract 6 Haralick-style properties via `graycoprops`
4. Average each property across the 4 angles → 1 scalar per property

### Properties

| Feature | High value indicates |
|---|---|
| `glcm_contrast` | Large local intensity variation → rough/uneven texture |
| `glcm_dissimilarity` | High neighbourhood intensity differences |
| `glcm_homogeneity` | Smoothly varying texture (inverse of contrast) |
| `glcm_energy` | Uniformity / regularity of texture pattern |
| `glcm_correlation` | Linear dependency between neighbouring pixels |
| `glcm_ASM` | Angular Second Moment — peaks for highly regular patterns (tiled roads) |

### Parameters
| Parameter | Value |
|---|---|
| Distances | [1] |
| Angles | 0°, 45°, 90°, 135° (averaged) |
| Levels | 256 |
| Symmetric | True |
| Normalised | True |

---

## Feature Group C — LBP Histogram (10 features)

**Column names:** `lbp_00` … `lbp_09`

### Method
1. Compute Local Binary Pattern (LBP) image using `skimage.feature.local_binary_pattern`
   with the **uniform** method
2. For uniform LBP with P=8, there are **P + 2 = 10** distinct codes
   (8 uniform patterns + 1 aggregated non-uniform bin)
3. Compute a 10-bin histogram, normalise to sum to 1.0

### Interpretation
- LBP captures micro-texture: flat regions, edges, spots, and corners
- Smooth road → mostly flat regions and uniform gradients (low-code bins)
- Rough road → many edge/corner patterns (high non-uniform code count)
- Tiled road → regular edge patterns from tile joints (periodic structure)

### Parameters
| Parameter | Value |
|---|---|
| P (neighbours) | 8 |
| R (radius) | 1 |
| Method | uniform |
| Histogram bins | 10 (P + 2) |
| Normalisation | Sum = 1.0 |
| Input | gray_blur (blur applied) |

---

## Feature Group D — Edge Density (1 feature)

**Column name:** `edge_density`

### Method
1. Apply Canny edge detector on `gray_raw` (no pre-blur; Canny's own σ=1.0 Gaussian is used)
2. Compute: `edge_density = number of edge pixels / total pixels`
3. Result is a scalar in [0.0, 1.0]

### Interpretation
| Class | Expected edge density |
|---|---|
| Smooth Road | Low — few sharp transitions on uniform asphalt |
| Rough Road | High — many edges from gravel, cracks, cobblestone gaps |
| Tiled/Paved Road | Medium-high — regular grid of tile joints |

### Parameters
| Parameter | Value |
|---|---|
| Sigma (Canny internal Gaussian) | 1.0 |
| Input | gray_raw (no external blur) |

---

## Complete Feature Vector Layout

```
Index   Name                Group   Semantic
------  ------------------  ------  -------------------------------------------
 0-15   hist_00 … hist_15   A       Intensity distribution (16 bins, sum=1)
16      glcm_contrast       B       Texture roughness
17      glcm_dissimilarity  B       Neighbourhood intensity difference
18      glcm_homogeneity    B       Texture smoothness
19      glcm_energy         B       Texture uniformity / regularity
20      glcm_correlation    B       Pixel neighbourhood linear correlation
21      glcm_ASM            B       Angular Second Moment
22-31   lbp_00 … lbp_09    C       Micro-texture code histogram (sum=1)
32      edge_density        D       Fraction of Canny edge pixels
```

---

## Design Decisions

- **No deep features, SIFT, ORB, or Gabor filters** — kept deliberately simple and
  interpretable for a Minimum Distance Classifier.
- **Only 33 features** — avoids the curse of dimensionality. With 600 training
  samples per class, 33 features is well within the safe regime.
- **Normalised histograms** — make the features invariant to absolute image size.
- **Single GLCM distance** — distance=1 is appropriate for 128×128 road texture
  images where the relevant texture scale is fine-grained.
- **Mild blur before GLCM/LBP** — JPEG compression artefacts can introduce
  spurious high-frequency texture; radius=1 eliminates this without smearing
  the real texture.
- **Canny uses no external blur** — Canny's built-in Gaussian (σ=1) is sufficient
  and preserves fine edge detail that would be lost by a second blur pass.
