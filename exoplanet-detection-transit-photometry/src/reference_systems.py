"""
Reference parameters for real, published systems.
 
Sources:
    - Kepler-10b:  Batalha et al. 2011
    - Kepler-90 system: Shallue & Vanderburg 2018 (AstroNet paper), Cabrera et al. 2014
    - TRAPPIST-1 system: Gillon et al. 2017, Agol et al. 2021 (refined ephemerides)
    - Kepler-452b: Jenkins et al. 2015
    - HAT-P-7b (hot Jupiter, for deep-transit regime): Pál et al. 2008
"""

# Confirmed-planet anchors: (period_days, Rp/Rstar, impact_parameter, Rstar_solar, Teff_K)
CONFIRMED_PLANET_ANCHORS = {
    "Kepler-10b":   dict(period=0.837,  rp_rs=0.0225, b=0.3,  rstar=1.065, teff=5708),
    "Kepler-90b":   dict(period=7.008,  rp_rs=0.0161, b=0.5,  rstar=1.20,  teff=6080),
    "Kepler-90i":   dict(period=14.449, rp_rs=0.0221, b=0.4,  rstar=1.20,  teff=6080),
    "TRAPPIST-1b":  dict(period=1.511,  rp_rs=0.0851, b=0.13, rstar=0.121, teff=2566),
    "TRAPPIST-1e":  dict(period=6.099,  rp_rs=0.0693, b=0.35, rstar=0.121, teff=2566),
    "Kepler-452b":  dict(period=384.8,  rp_rs=0.0138, b=0.6,  rstar=1.11,  teff=5757),
    "HAT-P-7b":     dict(period=2.205,  rp_rs=0.0813, b=0.49, rstar=1.84,  teff=6350),
}
 
# False-positive archetypes: what actually fools naive transit detection.
# Rp/Rs here is really "apparent depth ratio" for EBs (grazing / diluted).
FALSE_POSITIVE_ANCHORS = {
    # Grazing eclipsing binary: V-shaped, deep, short duration relative to depth
    "grazing_EB":        dict(period=3.2,  depth_ratio=0.09, shape="V", b=0.9),
    # Background eclipsing binary diluted by target star flux (common real FP mode)
    "background_EB":     dict(period=5.7,  depth_ratio=0.15, shape="U_diluted", b=0.2),
    # Secondary eclipse mimicking a transit (odd/even depth mismatch in reality)
    "secondary_eclipse":  dict(period=4.1,  depth_ratio=0.06, shape="U", b=0.1),
    # Short-period contact binary aliased to look like a shallow transit
    "aliased_contact_binary": dict(period=0.9, depth_ratio=0.04, shape="V", b=0.7),
}
 
# Typical Kepler long-cadence photometric precision (CDPP-like), by Kepler magnitude bin.
# ppm = parts per million, 6-hr combined differential photometric precision, approximate.
CDPP_BY_KEPMAG = {
    10: 20,     # bright star, e.g. HAT-P-7
    12: 40,
    14: 90,
    15: 150,
    16: 250,
}
 
KEPLER_LONG_CADENCE_MIN = 29.4  # minutes, real Kepler long-cadence sampling
