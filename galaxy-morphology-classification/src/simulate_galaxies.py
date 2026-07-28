"""
Procedural galaxy image simulator.
 
DISCLOSURE: These are SYNTHETIC images -- SDSS SkyServer, Galaxy Zoo,
Zenodo, and Kaggle are all unreachable from this sandbox (confirmed
host_not_allowed), so no real galaxy cutouts were downloaded. Images
are built from real, documented structural models (Sersic light
profiles, logarithmic spiral arms, PSF convolution, SDSS-like noise
characteristics -- see reference_morphology.py), aiming for realistic
MORPHOLOGY rather than pixel-for-pixel realism.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
 
from reference_morphology import SERSIC_INDEX, SDSS_IMAGING, SPIRAL_PITCH_ANGLE_DEG, MERGER_SEPARATION_RANGE_PX
 
SIZE = SDSS_IMAGING["image_size_px"]
 
 
def sersic_profile(size, n, r_eff, ellipticity=0.0, pa_deg=0.0, x0=None, y0=None, amplitude=1.0):
    """2D Sersic surface-brightness profile."""
    if x0 is None:
        x0 = size / 2
    if y0 is None:
        y0 = size / 2
    y, x = np.mgrid[0:size, 0:size]
    xr, yr = x - x0, y - y0
    theta = np.radians(pa_deg)
    xr_rot = xr * np.cos(theta) + yr * np.sin(theta)
    yr_rot = -xr * np.sin(theta) + yr * np.cos(theta)
    q = 1 - ellipticity
    r = np.sqrt(xr_rot ** 2 + (yr_rot / max(q, 0.1)) ** 2)
    # Sersic b_n approximation (Ciotti & Bertin 1999) for n>0.36
    b_n = 2 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n ** 2)
    profile = amplitude * np.exp(-b_n * ((r / max(r_eff, 0.5)) ** (1.0 / n) - 1))
    return profile
 
 
def spiral_arm_pattern(size, n_arms=2, pitch_deg=20, r_max=None, arm_width=1.8,
                        x0=None, y0=None, phase=0.0, amplitude=1.0):
    """Logarithmic spiral arm overdensity pattern, added on top of an
    exponential disk to create spiral structure."""
    if x0 is None:
        x0 = size / 2
    if y0 is None:
        y0 = size / 2
    if r_max is None:
        r_max = size / 2
    y, x = np.mgrid[0:size, 0:size]
    xr, yr = x - x0, y - y0
    r = np.sqrt(xr ** 2 + yr ** 2)
    phi = np.arctan2(yr, xr)
    b = 1.0 / np.tan(np.radians(pitch_deg))
 
    pattern = np.zeros((size, size))
    for k in range(n_arms):
        arm_phase = phase + 2 * np.pi * k / n_arms
        # logarithmic spiral: phi = arm_phase + b*ln(r) -> distance in phi-space
        log_r = np.log(np.clip(r, 1.0, None))
        delta_phi = (phi - (arm_phase + b * log_r) + np.pi) % (2 * np.pi) - np.pi
        arm_strength = np.exp(-(delta_phi ** 2) / (2 * (arm_width / 10) ** 2))
        radial_taper = np.exp(-r / (r_max * 0.6))
        pattern += arm_strength * radial_taper
    return amplitude * pattern
 
 
def apply_psf_and_noise(image, seed=None, psf_sigma_px=None, noise_level=None):
    rng = np.random.default_rng(seed)
    if psf_sigma_px is None:
        # convert PSF FWHM (arcsec) to pixel sigma using pixel scale
        fwhm_px = SDSS_IMAGING["psf_fwhm_arcsec"] / SDSS_IMAGING["pixel_scale_arcsec"]
        psf_sigma_px = fwhm_px / 2.3548
    blurred = gaussian_filter(image, sigma=psf_sigma_px)
    if noise_level is None:
        noise_level = rng.uniform(0.015, 0.04)
    noisy = blurred + rng.normal(0, noise_level, size=image.shape)
    return noisy, noise_level
 
 
def make_elliptical(seed=None):
    rng = np.random.default_rng(seed)
    r_eff = rng.uniform(6, 14)
    ell = rng.uniform(0.05, 0.5)
    pa = rng.uniform(0, 180)
    amp = rng.uniform(0.6, 1.0)
    img = sersic_profile(SIZE, SERSIC_INDEX["elliptical"], r_eff, ellipticity=ell, pa_deg=pa, amplitude=amp)
    return img
 
 
def make_spiral(seed=None):
    rng = np.random.default_rng(seed)
    bulge = sersic_profile(SIZE, SERSIC_INDEX["spiral_bulge"], rng.uniform(2, 5),
                            ellipticity=rng.uniform(0, 0.2), amplitude=rng.uniform(0.3, 0.6))
    disk_r = rng.uniform(10, 18)
    ell = rng.uniform(0.1, 0.55)  # inclination-driven ellipticity
    pa = rng.uniform(0, 180)
    disk = sersic_profile(SIZE, SERSIC_INDEX["spiral_disk"], disk_r, ellipticity=ell, pa_deg=pa,
                           amplitude=rng.uniform(0.3, 0.6))
    pitch = rng.uniform(*SPIRAL_PITCH_ANGLE_DEG)
    arms = spiral_arm_pattern(SIZE, n_arms=rng.choice([2, 2, 2, 3]), pitch_deg=pitch,
                               r_max=disk_r * 1.8, phase=rng.uniform(0, 2 * np.pi),
                               amplitude=rng.uniform(0.25, 0.5))
    # arms modulate the disk (ellipticity/rotation applied via disk footprint)
    img = bulge + disk * (1 + arms)
    return img
 
 
def make_merger(seed=None):
    rng = np.random.default_rng(seed)
    sep = rng.uniform(*MERGER_SEPARATION_RANGE_PX)
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = sep * np.cos(angle) / 2, sep * np.sin(angle) / 2
    cx, cy = SIZE / 2, SIZE / 2
 
    # two interacting galaxies -- mix of type for realism
    type1 = rng.choice(["elliptical", "spiral"])
    type2 = rng.choice(["elliptical", "spiral"])
 
    def component(x0, y0, gtype, seed_local):
        r = np.random.default_rng(seed_local)
        if gtype == "elliptical":
            return sersic_profile(SIZE, SERSIC_INDEX["elliptical"], r.uniform(5, 10),
                                   ellipticity=r.uniform(0.1, 0.4), pa_deg=r.uniform(0, 180),
                                   x0=x0, y0=y0, amplitude=r.uniform(0.5, 0.9))
        else:
            disk = sersic_profile(SIZE, SERSIC_INDEX["spiral_disk"], r.uniform(6, 10),
                                   ellipticity=r.uniform(0.1, 0.4), pa_deg=r.uniform(0, 180),
                                   x0=x0, y0=y0, amplitude=r.uniform(0.4, 0.7))
            return disk
 
    img1 = component(cx + dx, cy + dy, type1, seed)
    img2 = component(cx - dx, cy - dy, type2, seed + 1 if seed is not None else None)
    img = img1 + img2
 
    # tidal distortion: smear the combined image asymmetrically to mimic
    # tidal tails / disturbed morphology (a real, simple proxy, not a
    # full N-body merger simulation)
    shear_strength = rng.uniform(0.3, 0.9)
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    shear_field = shear_strength * np.sin((x - cx) / SIZE * np.pi * rng.uniform(1, 2))
    img_sheared = np.zeros_like(img)
    for row in range(SIZE):
        shift = int(shear_field[row].mean() * 3)
        img_sheared[row] = np.roll(img[row], shift)
    img = 0.6 * img + 0.4 * img_sheared
    return img
 
 
def build_dataset(n_per_class=350, seed0=2026):
    images, labels = [], []
    generators = {"elliptical": make_elliptical, "spiral": make_spiral, "merger": make_merger}
    seed = seed0
    for label, gen_fn in generators.items():
        for i in range(n_per_class):
            raw = gen_fn(seed=seed)
            raw = raw / (raw.max() + 1e-8)
            noisy, noise_level = apply_psf_and_noise(raw, seed=seed + 500000)
            images.append(noisy.astype(np.float32))
            labels.append(label)
            seed += 1
    return np.array(images), np.array(labels)
 
 
if __name__ == "__main__":
    images, labels = build_dataset()
    np.savez_compressed(
        "data/galaxy_images.npz",
        images=images, labels=labels,
    )
    print(f"Saved {images.shape[0]} images of shape {images.shape[1:]}")
    import collections
    print(collections.Counter(labels))
