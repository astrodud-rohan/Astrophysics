"""
Simulate a Gaia-like stellar catalog: multiple real open clusters plus a
field star population, with realistic photometric/astrometric noise.
 
Cluster distances/ages/proper motions are anchored to real published values (see
reference_data.py). Stellar physics (mass -> luminosity -> temperature ->
color/magnitude) uses SIMPLIFIED approximate analytic relations, not a
full stellar-evolution isochrone grid -- documented explicitly below so
this isn't mistaken for precision stellar modeling. The goal is to
reproduce realistic HR-diagram MORPHOLOGY (main sequence shape, turnoff
point moving with age, giant branch, photometric/astrometric error
inflation at faint magnitudes) for the purposes of demonstrating a
Gaia-style analysis pipeline (catalog querying pattern, HRD
construction, cluster membership recovery via clustering).
 
Approximate relations used (all documented, not claimed to be exact):
    - IMF: simplified Kroupa-like broken power law for mass sampling
    - Main sequence: L ~ M^3.5, R ~ M^0.8 (standard order-of-magnitude
      scaling relations for main-sequence stars, Msun=1)
    - Post-main-sequence: stars with mass such that main-sequence
      lifetime < cluster age are moved onto a simplified red giant
      branch (luminosity boosted, Teff dropped) -- a crude proxy for
      stellar evolution, not a real isochrone
    - Teff -> Gaia BP-RP color: approximate monotonic empirical fit
      calibrated to roughly match the published Gaia DR2/DR3 HRD
      morphology (Gaia Collaboration, Babusiaux et al. 2018), NOT an
      exact spectral synthesis result
    - Bolometric correction in G band: simple smooth function of Teff,
      peaking near zero for G dwarfs, more negative for hot/cool stars
"""
import numpy as np
import pandas as pd
 
from reference_data import (
    CLUSTERS, FIELD_POPULATION, GAIA_G_PRECISION_MMAG,
    GAIA_PARALLAX_PRECISION_MAS, GAIA_PM_PRECISION_MASYR, M_G_SUN,
)
 
RNG = np.random.default_rng(2026)
 
 
def sample_imf(n, m_min=0.15, m_max=8.0, seed=None):
    """Simplified Kroupa-like broken power law IMF sampling via inverse
    transform on a single power-law segment per mass regime (fast
    approximation, not a full piecewise-continuous IMF)."""
    rng = np.random.default_rng(seed)
    # Sample using a single power law with Kroupa-like high-mass slope
    # (alpha=2.3 for M>0.5, alpha=1.3 for M<0.5) via a simple mixture
    frac_low = 0.6
    n_low = int(n * frac_low)
    n_high = n - n_low
    u_low = rng.uniform(0, 1, n_low)
    alpha_low = 1.3
    m_low = (m_min ** (1 - alpha_low) + u_low * (0.5 ** (1 - alpha_low) - m_min ** (1 - alpha_low))) ** (1 / (1 - alpha_low))
    u_high = rng.uniform(0, 1, n_high)
    alpha_high = 2.3
    m_high = (0.5 ** (1 - alpha_high) + u_high * (m_max ** (1 - alpha_high) - 0.5 ** (1 - alpha_high))) ** (1 / (1 - alpha_high))
    return np.concatenate([m_low, m_high])
 
 
def main_sequence_lum_teff(mass_msun):
    """Simplified L(M), R(M) -> Teff(M) main-sequence relations."""
    L = mass_msun ** 3.5
    R = mass_msun ** 0.8
    # Stefan-Boltzmann: L = 4 pi R^2 sigma T^4  =>  T ~ (L/R^2)^0.25, normalized to Sun=5772K
    teff = 5772.0 * (L / R ** 2) ** 0.25
    return L, teff
 
 
def ms_lifetime_myr(mass_msun):
    """Main-sequence lifetime scaling, normalized to ~10 Gyr for the Sun."""
    return 10000.0 * mass_msun ** -2.5
 
 
def evolve_if_needed(mass_msun, age_myr, seed=None):
    """If a star's MS lifetime is shorter than the cluster age, move it
    onto a crude giant-branch proxy: boost luminosity, drop Teff."""
    rng = np.random.default_rng(seed)
    L, teff = main_sequence_lum_teff(mass_msun)
    lifetime = ms_lifetime_myr(mass_msun)
    evolved = age_myr > lifetime
    if np.isscalar(evolved):
        evolved = np.array([evolved])
    L = np.atleast_1d(L).astype(float)
    teff = np.atleast_1d(teff).astype(float)
    if evolved.any():
        boost = rng.uniform(20, 200, size=evolved.sum())
        L[evolved] *= boost
        teff[evolved] = rng.uniform(3800, 5000, size=evolved.sum())
    return L, teff, evolved
 
 
def teff_to_bp_rp(teff):
    """Approximate monotonic Teff -> Gaia BP-RP color fit, calibrated to
    roughly match published Gaia HRD morphology (NOT exact synthetic
    photometry)."""
    # Empirical-style inverse relation: hotter -> bluer (lower BP-RP)
    bp_rp = 8500.0 / teff - 1.15
    return bp_rp
 
 
def bolometric_correction_G(teff):
    """Simple smooth BC_G(Teff) proxy: near zero for G dwarfs (~5772K),
    increasingly negative away from that peak."""
    x = (np.log10(teff) - np.log10(5772.0))
    return -1.5 * x ** 2 * 10  # broad parabola in log-Teff space
 
 
def absolute_mag_G(L_lsun, teff):
    m_bol = 4.74 - 2.5 * np.log10(L_lsun)
    bc = bolometric_correction_G(teff)
    return m_bol - bc
 
 
def photometric_noise(mag, precision_table_mmag, seed=None):
    rng = np.random.default_rng(seed)
    mags_grid = sorted(precision_table_mmag)
    sigma_mmag = np.interp(mag, mags_grid, [precision_table_mmag[m] for m in mags_grid])
    sigma_mag = sigma_mmag / 1000.0
    return rng.normal(0, sigma_mag, size=np.shape(mag)), sigma_mag
 
 
def astrometric_noise(mag, precision_table, seed=None):
    rng = np.random.default_rng(seed)
    mags_grid = sorted(precision_table)
    sigma = np.interp(mag, mags_grid, [precision_table[m] for m in mags_grid])
    return rng.normal(0, sigma, size=np.shape(mag)), sigma
 
 
def simulate_cluster(name, params, seed):
    rng = np.random.default_rng(seed)
    n = params["n_members"]
    masses = sample_imf(n, seed=seed)
    L, teff, evolved = evolve_if_needed(masses, params["age_myr"], seed=seed + 1)
    bp_rp_true = teff_to_bp_rp(teff)
    M_G_true = absolute_mag_G(L, teff)
 
    parallax_true = rng.normal(params["parallax"], params["parallax_scatter"], n)
    parallax_true = np.clip(parallax_true, 0.1, None)
    dist_pc = 1000.0 / parallax_true
    apparent_G_true = M_G_true + 5 * np.log10(dist_pc) - 5
 
    pm_ra_true = rng.normal(params["pm_ra"], params["pm_scatter"], n)
    pm_dec_true = rng.normal(params["pm_dec"], params["pm_scatter"], n)
 
    g_noise, g_sigma = photometric_noise(apparent_G_true, GAIA_G_PRECISION_MMAG, seed=seed + 2)
    bprp_noise, bprp_sigma = photometric_noise(
        apparent_G_true, {k: v * 1.8 for k, v in GAIA_G_PRECISION_MMAG.items()}, seed=seed + 3
    )
    plx_noise, plx_sigma = astrometric_noise(apparent_G_true, GAIA_PARALLAX_PRECISION_MAS, seed=seed + 4)
    pmra_noise, pm_sigma = astrometric_noise(apparent_G_true, GAIA_PM_PRECISION_MASYR, seed=seed + 5)
    pmdec_noise, _ = astrometric_noise(apparent_G_true, GAIA_PM_PRECISION_MASYR, seed=seed + 6)
 
    df = pd.DataFrame({
        "source_id": [f"SIM-{name}-{i:04d}" for i in range(n)],
        "true_cluster": name,
        "mass_msun": masses,
        "evolved": evolved,
        "phot_g_mean_mag": apparent_G_true + g_noise,
        "bp_rp": bp_rp_true + bprp_noise,
        "parallax": parallax_true + plx_noise,
        "parallax_error": plx_sigma,
        "pmra": pm_ra_true + pmra_noise,
        "pmdec": pm_dec_true + pmdec_noise,
        "g_mag_error": g_sigma,
    })
    return df
 
 
def simulate_field(params, seed):
    rng = np.random.default_rng(seed)
    n = params["n_stars"]
    masses = sample_imf(n, m_min=0.1, m_max=3.0, seed=seed)
    field_ages = rng.uniform(500, 10000, n)  # broad age mix, Myr
    L, teff, evolved = evolve_if_needed(masses, field_ages, seed=seed + 1)
    bp_rp_true = teff_to_bp_rp(teff)
    M_G_true = absolute_mag_G(L, teff)
 
    log_parallax = rng.uniform(
        np.log10(params["parallax_min"]), np.log10(params["parallax_max"]), n
    )
    parallax_true = 10 ** log_parallax
    dist_pc = 1000.0 / parallax_true
    apparent_G_true = M_G_true + 5 * np.log10(dist_pc) - 5
 
    pm_ra_true = rng.normal(params["pm_ra_center"], params["pm_ra_spread"], n)
    pm_dec_true = rng.normal(params["pm_dec_center"], params["pm_dec_spread"], n)
 
    g_noise, g_sigma = photometric_noise(apparent_G_true, GAIA_G_PRECISION_MMAG, seed=seed + 2)
    bprp_noise, bprp_sigma = photometric_noise(
        apparent_G_true, {k: v * 1.8 for k, v in GAIA_G_PRECISION_MMAG.items()}, seed=seed + 3
    )
    plx_noise, plx_sigma = astrometric_noise(apparent_G_true, GAIA_PARALLAX_PRECISION_MAS, seed=seed + 4)
    pmra_noise, pm_sigma = astrometric_noise(apparent_G_true, GAIA_PM_PRECISION_MASYR, seed=seed + 5)
    pmdec_noise, _ = astrometric_noise(apparent_G_true, GAIA_PM_PRECISION_MASYR, seed=seed + 6)
 
    df = pd.DataFrame({
        "source_id": [f"SIM-FIELD-{i:05d}" for i in range(n)],
        "true_cluster": "field",
        "mass_msun": masses,
        "evolved": evolved,
        "phot_g_mean_mag": apparent_G_true + g_noise,
        "bp_rp": bp_rp_true + bprp_noise,
        "parallax": parallax_true + plx_noise,
        "parallax_error": plx_sigma,
        "pmra": pm_ra_true + pmra_noise,
        "pmdec": pm_dec_true + pmdec_noise,
        "g_mag_error": g_sigma,
    })
    return df
 
 
def build_catalog():
    dfs = []
    seed = 100
    for name, params in CLUSTERS.items():
        dfs.append(simulate_cluster(name, params, seed))
        seed += 1000
    dfs.append(simulate_field(FIELD_POPULATION, seed))
    catalog = pd.concat(dfs, ignore_index=True)
    # apply a faint-end magnitude cut, standard Gaia-like survey limit
    catalog = catalog[catalog["phot_g_mean_mag"] < 20.5].reset_index(drop=True)
    return catalog
 
 
if __name__ == "__main__":
    catalog = build_catalog()
    out_path = "../data/gaia_like_catalog.csv"
    catalog.to_csv(out_path, index=False)
    print(f"Saved {len(catalog)} stars -> {out_path}")
    print(catalog["true_cluster"].value_counts())