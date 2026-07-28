"""
Reference parameters anchoring the synthetic Gaia-like catalog to real,
published values.
 
Sources:
    - Pleiades: distance ~136 pc (parallax ~7.35 mas), age ~125 Myr
      (Cantat-Gaudin et al. 2018 Gaia DR2 cluster membership catalog;
      Gaia Collaboration parallax zero-point work)
    - Hyades: distance ~47 pc (parallax ~21.3 mas), age ~625-750 Myr
      (Perryman et al. 1998; Gaia DR2/DR3 Hyades membership studies)
    - NGC 752: distance ~440 pc (parallax ~2.27 mas), age ~1.4-2.0 Gyr
      (older open cluster, used to anchor a main-sequence turnoff further
      down than Pleiades/Hyades)
    - Main sequence color-absolute magnitude relation: approximate
      empirical relation for the region BP-RP in [-0.2, 3.0], calibrated
      to match roughly the Gaia DR2/DR3 published HR diagram morphology
      (Gaia Collaboration, Babusiaux et al. 2018, "Observational HRD")
    - Gaia DR3 photometric precision vs G magnitude and parallax
      precision vs G magnitude: approximate published scaling relations
      (Gaia DR3 documentation, Lindegren et al. 2021 astrometric
      performance tables), NOT exact per-source values
 
This file is the single documented source of "real-world anchoring."
All downstream simulation code samples noise/nuisance parameters around
these anchors -- it does not invent cluster properties from scratch.
"""
 
# name: (parallax_mas, parallax_scatter_mas [true depth/spread of cluster],
#         age_Myr, pm_ra_masyr, pm_dec_masyr, pm_scatter_masyr, n_members)
CLUSTERS = {
    "Pleiades": dict(parallax=7.35, parallax_scatter=0.15, age_myr=125,
                      pm_ra=19.9, pm_dec=-45.5, pm_scatter=0.8, n_members=250),
    "Hyades":   dict(parallax=21.3, parallax_scatter=0.6,  age_myr=680,
                      pm_ra=101.6, pm_dec=-27.7, pm_scatter=1.2, n_members=180),
    "NGC752":   dict(parallax=2.27, parallax_scatter=0.05, age_myr=1400,
                      pm_ra=9.7,   pm_dec=-11.7, pm_scatter=0.3, n_members=120),
}
 
# Field star population: broad parallax/PM distribution, much larger N,
# roughly matching a random Milky Way disk sightline mix
FIELD_POPULATION = dict(
    n_stars=3000,
    parallax_min=0.3,   # ~3.3 kpc
    parallax_max=15.0,  # ~67 pc (nearby field stars)
    pm_ra_center=0.0, pm_ra_spread=8.0,
    pm_dec_center=-5.0, pm_dec_spread=8.0,
)
 
# Gaia DR3 approximate photometric precision (mmag) vs G magnitude
# (bright end is photon-noise-floor-limited by calibration systematics,
# faint end scales steeply with flux)
GAIA_G_PRECISION_MMAG = {
    6: 0.2, 10: 0.3, 13: 0.5, 15: 1.5, 17: 5.0, 19: 20.0, 20: 40.0,
}
 
# Gaia DR3 approximate parallax precision (micro-arcsec -> converted to mas here)
# vs G magnitude, from published astrometric performance curves
GAIA_PARALLAX_PRECISION_MAS = {
    6: 0.02, 10: 0.03, 13: 0.05, 15: 0.08, 17: 0.20, 19: 0.7, 20: 1.3,
}
 
GAIA_PM_PRECISION_MASYR = {
    6: 0.02, 10: 0.03, 13: 0.06, 15: 0.10, 17: 0.25, 19: 0.9, 20: 1.6,
}
 
# Solar absolute magnitude in Gaia G band (approximate, standard reference value)
M_G_SUN = 4.67
