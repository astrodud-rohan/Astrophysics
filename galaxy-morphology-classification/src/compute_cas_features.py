"""
Classical quantitative morphology features: CAS (Concentration,
Asymmetry, Smoothness -- Conselice 2003) and Gini/M20 (Lotz et al.
2004). These are the ACTUAL pre-deep-learning feature set used in the
real morphology classification literature, computed here as a baseline
to compare against a CNN.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import rotate, uniform_filter
 
DATA_PATH = "data/galaxy_images.npz"
OUT_PATH = "data/cas_features.csv"
 
 
def flux_centroid(img):
    size = img.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    flux = np.clip(img, 0, None)
    total = flux.sum()
    if total <= 0:
        return size / 2, size / 2
    cx = (flux * x).sum() / total
    cy = (flux * y).sum() / total
    return cx, cy
 
 
def aperture_mask(shape, cx, cy, radius):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return r <= radius
 
 
def petrosian_like_radius(img, cx, cy, f=0.8):
    """Radius enclosing fraction f of total flux, centered on the flux
    centroid (not the geometric image center) -- matches real practice."""
    size = img.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    flux = np.clip(img, 0, None)
    total = flux.sum()
    if total <= 0:
        return size / 4
    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    cumflux = np.cumsum(flux.ravel()[order]) / total
    return np.interp(f, cumflux, r_sorted)
 
 
def concentration(img, f_inner=0.3, f_outer=0.8):
    """Ratio of circular radii containing f_outer vs f_inner of total
    flux, centered on the flux centroid (simplified Petrosian-radius-
    based concentration index, Conselice 2003 / Bershady et al. 2000)."""
    cx, cy = flux_centroid(img)
    r_inner = petrosian_like_radius(img, cx, cy, f_inner)
    r_outer = petrosian_like_radius(img, cx, cy, f_outer)
    return 5 * np.log10(max(r_outer, 1e-3) / max(r_inner, 1e-3))
 
 
def asymmetry(img, aperture_factor=1.5):
    """180-degree rotation asymmetry about the flux centroid, restricted
    to a compact aperture (Petrosian-like radius x aperture_factor) so
    the statistic isn't swamped by background noise over the full frame
    -- matches real CAS methodology (Conselice 2003)."""
    cx, cy = flux_centroid(img)
    r80 = petrosian_like_radius(img, cx, cy, 0.8)
    mask = aperture_mask(img.shape, cx, cy, r80 * aperture_factor)
 
    size = img.shape[0]
    shift_y, shift_x = size / 2 - cy, size / 2 - cx
    img_centered = np.roll(np.roll(img, int(round(shift_y)), axis=0), int(round(shift_x)), axis=1)
    mask_centered = np.roll(np.roll(mask, int(round(shift_y)), axis=0), int(round(shift_x)), axis=1)
    rotated = rotate(img_centered, 180, reshape=False, mode="constant")
 
    bg = np.percentile(img[~mask], 50) if (~mask).sum() > 10 else np.percentile(img, 10)
    diff = np.abs((img_centered - bg) - (rotated - bg)) * mask_centered
    denom = (np.abs(img_centered - bg) * mask_centered).sum()
    return diff.sum() / max(denom, 1e-8)
 
 
def smoothness(img, boxcar_frac=0.15, aperture_factor=1.5):
    """Residual from a boxcar-smoothed version within a compact aperture
    -- captures small-scale clumpiness (high for irregular/merging
    systems), background-subtracted (Conselice 2003)."""
    cx, cy = flux_centroid(img)
    r80 = petrosian_like_radius(img, cx, cy, 0.8)
    mask = aperture_mask(img.shape, cx, cy, r80 * aperture_factor)
 
    size = img.shape[0]
    box = max(int(size * boxcar_frac), 3)
    smoothed = uniform_filter(img, size=box)
    bg = np.percentile(img[~mask], 50) if (~mask).sum() > 10 else np.percentile(img, 10)
    diff = np.abs((img - bg) - (smoothed - bg)) * mask
    denom = (np.abs(img - bg) * mask).sum()
    return diff.sum() / max(denom, 1e-8)
 
 
def gini_coefficient(img):
    flux = np.clip(img, 0, None).ravel()
    flux = np.sort(flux)
    n = len(flux)
    if flux.sum() == 0:
        return 0.0
    cum = np.cumsum(flux)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
 
 
def m20(img):
    """Normalized second-order moment of the brightest 20% of flux,
    centered on the flux centroid (Lotz et al. 2004)."""
    size = img.shape[0]
    cx, cy = flux_centroid(img)
    y, x = np.mgrid[0:size, 0:size]
    flux = np.clip(img, 0, None)
    total_flux = flux.sum()
    if total_flux <= 0:
        return 0.0
    mtot = (flux * ((x - cx) ** 2 + (y - cy) ** 2)).sum()
 
    flat = flux.ravel()
    order = np.argsort(flat)[::-1]
    cumflux = np.cumsum(flat[order])
    cutoff_idx = np.searchsorted(cumflux, 0.2 * total_flux)
    bright_mask = np.zeros_like(flat, dtype=bool)
    bright_mask[order[:cutoff_idx + 1]] = True
    bright_mask = bright_mask.reshape(img.shape)
 
    m_bright = (flux * ((x - cx) ** 2 + (y - cy) ** 2) * bright_mask).sum()
    if mtot <= 0 or m_bright <= 0:
        return 0.0
    return np.log10(m_bright / mtot)
 
 
def main():
    d = np.load(DATA_PATH, allow_pickle=True)
    images, labels = d["images"], d["labels"]
 
    rows = []
    for i in range(len(images)):
        img = images[i]
        rows.append(dict(
            label=labels[i],
            concentration=concentration(img),
            asymmetry=asymmetry(img),
            smoothness=smoothness(img),
            gini=gini_coefficient(img),
            m20=m20(img),
        ))
        if (i + 1) % 300 == 0:
            print(f"  processed {i+1}/{len(images)}")
 
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(df.groupby("label")[["concentration", "asymmetry", "smoothness", "gini", "m20"]].mean())
 
 
if __name__ == "__main__":
    main()
