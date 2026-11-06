"""
features.py
===========
Feature-extraction library for Road Surface Condition Classification.

All public functions are deterministic (no randomness).
All functions accept a file path (str | Path) or a pre-loaded numpy uint8
grayscale array.

Pipeline overview
-----------------
preprocess_image()
    Load RGB → crop top-50% sky/horizon/vehicles → resize to TARGET_SIZE
    → convert to grayscale → optional mild Gaussian blur

Feature groups
--------------
A. Grayscale histogram  — 16 normalised bins
B. GLCM texture         — 6 properties, distance=1, 4 angles, averaged
C. LBP histogram        — 10 uniform-pattern bins (P=8, R=1), normalised
D. Edge density         — Canny edge pixel ratio (scalar)

Total feature vector length: 16 + 6 + 10 + 1 = 33 features
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern

# ── Pre-processing constants ──────────────────────────────────────────────────
# Final grayscale size fed into all feature extractors.
# 128×128 squares the image consistently regardless of input aspect ratio.
TARGET_SIZE: tuple[int, int] = (128, 128)   # (width, height) for PIL

# Fraction of rows to remove from the top (sky / horizon / vehicles band).
# Visual inspection (crop_comparison.png) showed top-30% still kept cars,
# buildings, and vegetation in mid-frame.  top-50% retains only the near-road
# surface area, giving texture features that truly represent road condition.
CROP_TOP_FRACTION: float = 0.50

# Gaussian blur kernel radius applied before GLCM and LBP extraction.
# 0 → no blur.  1 → PIL GaussianBlur radius=1 (very mild).
BLUR_RADIUS: float = 1.0

# ── GLCM constants ────────────────────────────────────────────────────────────
GLCM_DISTANCES = [1]                            # 1-pixel offset for small images
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]   # 0°, 45°, 90°, 135°
GLCM_PROPS     = ["contrast", "dissimilarity", "homogeneity",
                  "energy", "correlation", "ASM"]
GLCM_LEVELS    = 256

# ── LBP constants ─────────────────────────────────────────────────────────────
LBP_P      = 8          # number of circularly symmetric neighbours
LBP_R      = 1          # radius
LBP_METHOD = "uniform"  # uniform LBP gives P+2 = 10 distinct codes
LBP_BINS   = LBP_P + 2  # = 10

# ── Histogram constants ───────────────────────────────────────────────────────
HIST_BINS = 16


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    img_path: str | Path,
    blur_radius: float = BLUR_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and pre-process a road image.

    Returns
    -------
    gray_raw : np.ndarray uint8, shape TARGET_SIZE[1] × TARGET_SIZE[0]
        Grayscale image without blur (used for histogram & Canny).
    gray_blur : np.ndarray uint8, same shape
        Grayscale image with mild Gaussian blur (used for GLCM & LBP).
    """
    # Load RGB
    img_pil = Image.open(img_path).convert("RGB")

    w, h = img_pil.size
    # Crop top CROP_TOP_FRACTION to remove sky/horizon
    top_crop = int(h * CROP_TOP_FRACTION)
    img_pil  = img_pil.crop((0, top_crop, w, h))   # left, upper, right, lower

    # Resize to TARGET_SIZE using high-quality Lanczos resampling
    img_pil = img_pil.resize(TARGET_SIZE, Image.LANCZOS)

    # Convert to grayscale
    gray_pil = img_pil.convert("L")
    gray_raw = np.array(gray_pil, dtype=np.uint8)

    # Mild Gaussian blur for texture / LBP extraction
    if blur_radius > 0:
        from PIL import ImageFilter
        gray_blur = np.array(
            gray_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
            dtype=np.uint8,
        )
    else:
        gray_blur = gray_raw.copy()

    return gray_raw, gray_blur


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def extract_histogram(gray: np.ndarray) -> np.ndarray:
    """
    A. Grayscale histogram — 16 normalised bins.

    Parameters
    ----------
    gray : uint8 ndarray (H, W)

    Returns
    -------
    hist : float64 ndarray, shape (16,)  — sums to 1.0
    """
    hist, _ = np.histogram(gray, bins=HIST_BINS, range=(0, 256))
    total   = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        hist = np.zeros(HIST_BINS, dtype=np.float64)
    return hist


def extract_glcm(gray: np.ndarray) -> np.ndarray:
    """
    B. GLCM texture — 6 properties, distance=1, 4 angles, mean across angles.

    Parameters
    ----------
    gray : uint8 ndarray (H, W)

    Returns
    -------
    features : float64 ndarray, shape (6,)
        [contrast, dissimilarity, homogeneity, energy, correlation, ASM]
    """
    glcm = graycomatrix(
        gray,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    # glcm shape: (levels, levels, len(distances), len(angles))
    feats = []
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)   # shape (len(distances), len(angles))
        feats.append(float(vals.mean()))
    return np.array(feats, dtype=np.float64)


def extract_lbp(gray: np.ndarray) -> np.ndarray:
    """
    C. LBP histogram — 10 uniform-pattern bins, normalised.

    Parameters
    ----------
    gray : uint8 ndarray (H, W)  — should be the blur version

    Returns
    -------
    hist : float64 ndarray, shape (10,)  — sums to 1.0
    """
    lbp  = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS))
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        hist = np.zeros(LBP_BINS, dtype=np.float64)
    return hist


def extract_edge_density(gray: np.ndarray) -> float:
    """
    D. Edge density — Canny edge pixel count / total pixels.

    Parameters
    ----------
    gray : uint8 ndarray (H, W)

    Returns
    -------
    density : float64 scalar in [0, 1]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edges = canny(gray.astype(np.float64), sigma=1.0)
    return float(edges.sum()) / float(edges.size)


# ─────────────────────────────────────────────────────────────────────────────
# Combined extractor
# ─────────────────────────────────────────────────────────────────────────────

# Column names for the complete feature vector — in extraction order
FEATURE_NAMES: list[str] = (
    [f"hist_{i:02d}"   for i in range(HIST_BINS)]   # hist_00 … hist_15  (16)
    + [f"glcm_{p}"     for p in GLCM_PROPS]          # glcm_contrast …    (6)
    + [f"lbp_{i:02d}"  for i in range(LBP_BINS)]     # lbp_00 … lbp_09   (10)
    + ["edge_density"]                                # edge_density        (1)
)   # total: 33 features


def extract_all(img_path: str | Path) -> Optional[dict]:
    """
    Run the full extraction pipeline on one image.

    Returns
    -------
    dict  with keys matching FEATURE_NAMES, or None on failure.
    """
    try:
        gray_raw, gray_blur = preprocess_image(img_path)

        hist  = extract_histogram(gray_raw)
        glcm  = extract_glcm(gray_blur)
        lbp   = extract_lbp(gray_blur)
        edge  = extract_edge_density(gray_raw)

        values = np.concatenate([hist, glcm, lbp, [edge]])
        return dict(zip(FEATURE_NAMES, values))

    except Exception as exc:
        # Let the caller decide how to handle / log the failure
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers (imported by extract_features.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_visualisation(
    img_path: str | Path,
    save_path: str | Path,
    class_name: str,
    image_id:   str,
) -> None:
    """
    Save a 2×3 panel figure for one image:
      [original RGB | cropped+resized gray | Canny edges]
      [LBP image    | Grayscale histogram  | LBP histogram]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.feature import local_binary_pattern, canny as skimage_canny

    # ── Preprocessing ─────────────────────────────────────────────────────────
    img_pil_orig = Image.open(img_path).convert("RGB")
    w, h         = img_pil_orig.size
    top_crop     = int(h * CROP_TOP_FRACTION)
    img_cropped  = img_pil_orig.crop((0, top_crop, w, h))
    img_resized  = img_cropped.resize(TARGET_SIZE, Image.LANCZOS)
    gray_raw     = np.array(img_resized.convert("L"), dtype=np.uint8)

    from PIL import ImageFilter
    gray_blur    = np.array(
        img_resized.convert("L").filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)),
        dtype=np.uint8,
    )

    edges = skimage_canny(gray_raw.astype(np.float64), sigma=1.0)
    lbp   = local_binary_pattern(gray_blur, P=LBP_P, R=LBP_R, method=LBP_METHOD)

    hist_vals, hist_edges = np.histogram(gray_raw, bins=HIST_BINS, range=(0, 256))
    hist_vals = hist_vals / hist_vals.sum()

    lbp_hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS))
    lbp_hist    = lbp_hist / lbp_hist.sum()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(f"{class_name}  |  {image_id}", fontsize=11, fontweight="bold")

    # Row 0
    axes[0, 0].imshow(np.array(img_resized))
    axes[0, 0].set_title("Original (resized RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray_raw, cmap="gray")
    axes[0, 1].set_title(f"Grayscale\n(crop top {int(CROP_TOP_FRACTION*100)}%, resize {TARGET_SIZE[0]}×{TARGET_SIZE[1]})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(edges, cmap="gray")
    edge_dens = edges.sum() / edges.size
    axes[0, 2].set_title(f"Canny Edges\n(density = {edge_dens:.4f})")
    axes[0, 2].axis("off")

    # Row 1
    axes[1, 0].imshow(lbp, cmap="nipy_spectral",
                      vmin=0, vmax=LBP_BINS)
    axes[1, 0].set_title(f"LBP (P={LBP_P}, R={LBP_R}, uniform)")
    axes[1, 0].axis("off")

    bin_centres = (hist_edges[:-1] + hist_edges[1:]) / 2
    axes[1, 1].bar(bin_centres, hist_vals,
                   width=(hist_edges[1] - hist_edges[0]) * 0.9,
                   color="steelblue", edgecolor="white")
    axes[1, 1].set_title("Grayscale Histogram (16 bins)")
    axes[1, 1].set_xlabel("Pixel intensity")
    axes[1, 1].set_ylabel("Normalised frequency")

    axes[1, 2].bar(range(LBP_BINS), lbp_hist,
                   color="darkorange", edgecolor="white")
    axes[1, 2].set_title(f"LBP Histogram ({LBP_BINS} uniform bins)")
    axes[1, 2].set_xlabel("LBP code")
    axes[1, 2].set_ylabel("Normalised frequency")
    axes[1, 2].set_xticks(range(LBP_BINS))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
