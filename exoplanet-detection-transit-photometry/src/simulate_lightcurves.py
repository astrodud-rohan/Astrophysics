"""
Light curve simulator for the exoplanet detection pipeline.
 
DISCLOSURE: Signals are anchored to real published system parameters
(see reference_systems.py) and combined with:
    - batman-package transit models (Mandel & Agol 2002 limb-darkened
      transit light curves)
    - stellar granulation + spot-rotation variability (Gaussian
      Process with a quasi-periodic kernel, standard in the
      Kepler/TESS variability-modeling literature)
    - Kepler long-cadence sampling (29.4 min) and realistic
      photon+read noise scaled by CDPP tables
 
This mirrors the standard "injection-recovery" testing methodology
used by the Kepler/TESS pipelines themselves to validate detection
efficiency and false-alarm rates.
"""
import numpy as np
import batman
 
from reference_systems import (
    CONFIRMED_PLANET_ANCHORS,
    FALSE_POSITIVE_ANCHORS,
    CDPP_BY_KEPMAG,
    KEPLER_LONG_CADENCE_MIN,
)
 
RNG = np.random.default_rng(42)
 
 
def kepler_time_array(baseline_days=90.0):
    """Kepler long-cadence sampling grid."""
    dt = KEPLER_LONG_CADENCE_MIN / (60 * 24)  # days
    return np.arange(0, baseline_days, dt)
 
 
def quasi_periodic_gp_variability(t, amplitude=0.0015, period=None, length_scale=15.0, seed=None):
    """
    Approximate stellar rotation/granulation variability using a
    quasi-periodic covariance kernel, sampled via a truncated Fourier
    approach (fast, avoids full GP matrix inversion for long baselines).
 
    This is a standard proxy for the quasi-periodic GP kernel
    (e.g. celerite / George rotation kernels) used in real stellar
    variability modeling.
    """
    rng = np.random.default_rng(seed)
    if period is None:
        period = rng.uniform(5, 30)  # days, typical stellar rotation period range
    n_harmonics = 4
    signal = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        decay = np.exp(-((k - 1) ** 2) / (2 * (length_scale / period) ** 2))
        signal += (amplitude / k) * decay * np.sin(2 * np.pi * k * t / period + phase)
    # slow drift component (instrumental / long-term granulation)
    drift = rng.normal(0, amplitude * 0.3) * np.sin(2 * np.pi * t / (rng.uniform(40, 80)))
    return signal + drift
 
 
def photometric_noise(t, kepmag=None, seed=None):
    rng = np.random.default_rng(seed)
    if kepmag is None:
        kepmag = rng.choice(list(CDPP_BY_KEPMAG.keys()))
    # interpolate CDPP (ppm) for this magnitude
    mags = sorted(CDPP_BY_KEPMAG)
    cdpp_ppm = np.interp(kepmag, mags, [CDPP_BY_KEPMAG[m] for m in mags])
    sigma = cdpp_ppm * 1e-6
    return rng.normal(0, sigma, size=len(t)), sigma
 
 
def batman_transit_flux(t, period, rp_rs, b, rstar_solar, t0=None, u=(0.3, 0.2), seed=None):
    """Generate a limb-darkened transit model using batman-package."""
    rng = np.random.default_rng(seed if seed is not None else int(period * 1000) % 2**32)
    if t0 is None:
        t0 = rng.uniform(0, period)
 
    # Semi-major axis in stellar radii via Kepler's third law approx
    # (assumes ~solar-mass host; adequate for simulation purposes)
    a_rstar = 215.0 * (period / 365.25) ** (2 / 3) / rstar_solar
 
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = max(a_rstar, 2.0)
    params.inc = np.degrees(np.arccos(b / max(a_rstar, 2.0)))
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(u)
    params.limb_dark = "quadratic"
 
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)
    return flux, dict(t0=t0, period=period, rp_rs=rp_rs, b=b, a_rstar=a_rstar)
 
 
def synthetic_eb_flux(t, period, depth_ratio, shape, b, t0=None, seed=None):
    """
    Approximate eclipsing-binary-like flux dips that are NOT physical
    transits but are shaped to resemble the false-positive archetypes
    that fool naive box-search detection (V-shaped grazing eclipses,
    diluted U-shapes, etc.), built from simple geometric eclipse models
    rather than full physical EB light curve synthesis.
    """
    rng = np.random.default_rng(seed if seed is not None else int(period * 777) % 2**32)
    if t0 is None:
        t0 = rng.uniform(0, period)
    phase = ((t - t0 + period / 2) % period) - period / 2
    dur = period * rng.uniform(0.02, 0.06) * (1.3 if shape == "V" else 1.0)
 
    flux = np.ones_like(t)
    in_eclipse = np.abs(phase) < dur / 2
    x = np.clip(np.abs(phase[in_eclipse]) / (dur / 2), 0, 1)
 
    if shape == "V":
        depth_profile = depth_ratio * (1 - x)  # linear V-shape (grazing)
    elif shape == "U_diluted":
        depth_profile = depth_ratio * np.sqrt(1 - x**2) * 0.6  # diluted, flatter bottom
    else:  # "U"
        depth_profile = depth_ratio * np.sqrt(np.clip(1 - x**2, 0, 1))
 
    flux[in_eclipse] -= depth_profile
    # secondary eclipse (real EBs show one at phase ~0.5) — key vetting signature
    phase2 = ((t - (t0 + period / 2) + period / 2) % period) - period / 2
    in_sec = np.abs(phase2) < dur / 2
    x2 = np.clip(np.abs(phase2[in_sec]) / (dur / 2), 0, 1)
    flux[in_sec] -= depth_ratio * 0.35 * np.sqrt(np.clip(1 - x2**2, 0, 1))
 
    return flux, dict(t0=t0, period=period, depth_ratio=depth_ratio, shape=shape, dur=dur)
 
 
def make_lightcurve(label, anchor_name, baseline_days=90.0, seed=None):
    """
    Build one full synthetic light curve: baseline=1 + variability +
    (transit OR EB signal OR nothing, per label) + photometric noise.
 
    label in {"planet", "false_positive", "no_signal"}
    """
    rng_seed = seed if seed is not None else RNG.integers(0, 2**32 - 1)
    rng = np.random.default_rng(rng_seed)
    t = kepler_time_array(baseline_days)
 
    kepmag = rng.uniform(10, 16)
    noise, sigma = photometric_noise(t, kepmag=kepmag, seed=rng_seed)
    variability = quasi_periodic_gp_variability(
        t, amplitude=rng.uniform(0.0003, 0.003), seed=rng_seed + 1
    )
 
    flux = np.ones_like(t)
    meta = dict(kepmag=kepmag, sigma=sigma, anchor=anchor_name, label=label)
 
    if label == "planet":
        p = CONFIRMED_PLANET_ANCHORS[anchor_name]
        # jitter anchor params slightly so we're not just replaying one exact system
        period = p["period"] * rng.uniform(0.97, 1.03)
        rp_rs = p["rp_rs"] * rng.uniform(0.9, 1.1)
        b = np.clip(p["b"] * rng.uniform(0.85, 1.15), 0, 0.95)
        tr_flux, tmeta = batman_transit_flux(t, period, rp_rs, b, p["rstar"], seed=rng_seed + 2)
        flux *= tr_flux
        meta.update(tmeta)
 
    elif label == "false_positive":
        p = FALSE_POSITIVE_ANCHORS[anchor_name]
        period = p["period"] * rng.uniform(0.95, 1.05)
        depth = p["depth_ratio"] * rng.uniform(0.85, 1.15)
        eb_flux, emeta = synthetic_eb_flux(
            t, period, depth, p["shape"], p["b"], seed=rng_seed + 3
        )
        flux *= eb_flux
        meta.update(emeta)
 
    # label == "no_signal": flux stays at baseline (pure variability + noise)
 
    flux = flux * (1 + variability) + noise
    return t, flux, meta
